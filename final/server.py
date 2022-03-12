import re
import numpy as np
import pandas as pd
from string import punctuation
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask import Flask, request, render_template
import warnings

warnings.filterwarnings("ignore")


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    """
    Функция очистки текста. В качестве опции удаление стопслов и
    привидение слов в начальную форму
    """
    text = text.lower().split()
    # Опционально, удаление стоп слов
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = [w for w in text if not w in stop_words]
    text = " ".join(text)
    # Удаление пунктуации
    text = "".join([c for c in text if c not in punctuation])
    # Очистка текста
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Опционально, привидение слов к начальном форме
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    return text


def fq1_freq(row, q_dict):
    """
    Функция расчета частотности question1
    """
    return len(q_dict[row['question1']])


def fq2_freq(row, q_dict):
    """
    Функция расчета частотности question2
    """
    return len(q_dict[row['question2']])


def fq1_q2_intersect(row, q_dict):
    """
    Функция расчета пересечений
    """
    return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))


app = Flask(__name__)


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/data', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form

        text_1 = text_to_wordlist(form_data['q1'], remove_stopwords=False, stem_words=False)
        text_2 = text_to_wordlist(form_data['q2'], remove_stopwords=False, stem_words=False)

        sequences_1 = tokenizer.texts_to_sequences(text_1)
        sequences_2 = tokenizer.texts_to_sequences(text_2)

        data_1 = pad_sequences(sequences_1, maxlen=max_sequence_length)
        data_2 = pad_sequences(sequences_2, maxlen=max_sequence_length)

        q_dict[form_data['q1']].add(form_data['q2'])
        q_dict[form_data['q2']].add(form_data['q1'])

        q1_q2_intersect = len(set(q_dict[form_data['q1']]).intersection(set(q_dict[form_data['q2']])))
        q1_freq = len(q_dict[form_data['q1']])
        q2_freq = len(q_dict[form_data['q2']])

        data_leaks = [[q1_q2_intersect, q1_freq, q2_freq]]
        data_leaks = ss.transform(data_leaks)

        predictions = model.predict([data_1[:1], data_2[:1], data_leaks[:1]], verbose=1)
        predictions += model.predict([data_2[:1], data_1[:1], data_leaks[:1]], verbose=1)
        predictions /= 2

        return render_template('data.html', form_data=predictions)


if __name__ == '__main__':
    PATH = ''
    PATH_DATA = PATH + 'data/'
    embedding_file = PATH_DATA + 'glove.840B.300d.txt'
    train_data_file = PATH_DATA + 'train.csv'
    test_data_file = PATH_DATA + 'test.csv'
    max_sequence_length = 60
    max_num_words = 200_000
    embedding_dim = 300
    validation_split_ratio = 0.1
    num_lstm = 231
    num_dense = 144
    rate_drop_lstm = 0.33
    rate_drop_dense = 0.38
    lstm_name = 'lstm_{:d}_{:d}_{:.2f}_{:.2f}'.format(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)
    act_f = 'relu'
    re_weight = True
    np.random.seed(21)

    print(f'Load LSTM model - {lstm_name}')

    embeddings_index = {}
    f = open(embedding_file, encoding='utf-8')

    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print(f'Found {len(embeddings_index)} word vectors.')

    train_texts_1 = []
    train_texts_2 = []
    train_labels = []

    df_train = pd.read_csv(train_data_file, encoding='utf-8')
    df_train = df_train.fillna('empty')
    train_q1 = df_train['question1'].values
    train_q2 = df_train['question2'].values
    train_labels = df_train['is_duplicate'].values

    for tr_text in train_q1:
        train_texts_1.append(text_to_wordlist(tr_text, remove_stopwords=False, stem_words=False))

    for tr_text in train_q2:
        train_texts_2.append(text_to_wordlist(tr_text, remove_stopwords=False, stem_words=False))

    print(f'{len(train_texts_1)} texts are found in train.csv')

    test_texts_1 = []
    test_texts_2 = []
    test_ids = []

    df_test = pd.read_csv(test_data_file, encoding='utf-8', low_memory=False)
    df_test = df_test.fillna('empty')
    test_q1 = df_test['question1'].values
    test_q2 = df_test['question2'].values
    test_ids = df_test['test_id'].values

    for te_text in test_q1:
        test_texts_1.append(text_to_wordlist(te_text, remove_stopwords=False, stem_words=False))

    for te_text in test_q2:
        test_texts_2.append(text_to_wordlist(te_text, remove_stopwords=False, stem_words=False))

    print(f'{len(test_texts_1)} texts are found in test.csv')

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(train_texts_1 + train_texts_2 + test_texts_1 + test_texts_2)

    train_sequences_1 = tokenizer.texts_to_sequences(train_texts_1)
    train_sequences_2 = tokenizer.texts_to_sequences(train_texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    word_index = tokenizer.word_index
    print(f'{len(word_index)} unique tokens are found')

    # pad all train with Max_Sequence_Length
    train_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    print(f'Shape of train data tensor: {train_data_1.shape}')
    print(f'Shape of train labels tensor: {train_labels.shape}')

    # pad all test with Max_Sequence_Length
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)
    print(f'Shape of test data tensor: {test_data_2.shape}')
    print(f'Shape of test ids tensor:{test_ids.shape}')

    questions = pd.concat([df_train[['question1', 'question2']], \
                           df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')

    q_dict = defaultdict(set)
    for i in range(questions.shape[0]):
        q_dict[questions.question1[i]].add(questions.question2[i])
        q_dict[questions.question2[i]].add(questions.question1[i])

    df_train['q1_q2_intersect'] = df_train.apply(lambda x: fq1_q2_intersect(x, q_dict), axis=1)
    df_train['q1_freq'] = df_train.apply(lambda x: fq1_freq(x, q_dict), axis=1)
    df_train['q2_freq'] = df_train.apply(lambda x: fq2_freq(x, q_dict), axis=1)

    df_test['q1_q2_intersect'] = df_test.apply(lambda x: fq1_q2_intersect(x, q_dict), axis=1)
    df_test['q1_freq'] = df_test.apply(lambda x: fq1_freq(x, q_dict), axis=1)
    df_test['q2_freq'] = df_test.apply(lambda x: fq2_freq(x, q_dict), axis=1)

    train_leaks = df_train[['q1_q2_intersect', 'q1_freq', 'q2_freq']]
    test_leaks = df_test[['q1_q2_intersect', 'q1_freq', 'q2_freq']]

    ss = StandardScaler()
    ss.fit(np.vstack((train_leaks, test_leaks)))

    model = load_model('result/model')
    print('Ready')
    app.run('0.0.0.0', port=9000, debug=True)
