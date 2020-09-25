#!/usr/bin/env python
# coding: utf-8

# Импортируем библиотеку Numpy
import numpy as np

# Определяем функцию подсчета результата
def game_score(game_core):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число
    Функция принимает функцию, которая реализует игру.
    Функция возвращает среднее количествово попыток.
    '''
    count_list = []
    np.random.seed(42)  # фиксируем RANDOM SEED, чтобы эксперимент был воспроизводим
    random_numbers = np.random.randint(1,101, size=(1000))
    
    for number in random_numbers:
        count_list.append(game_core(number))
    score = int(np.mean(count_list))
    
    return(score)

# Определяем функцию игры
def game_core(number):
    '''Устанавливаем минимальное и максимальное возможное значение.
    На каждом этапе уменьшаем диапазон в 2 раза с учетом ответа больше/меньше.
    Функция принимает загаданное число и возвращает число попыток.
    '''
    count = 1
    min_value = 1
    max_value = 100
    predict = 50
    
    while predict != number:
        if number > predict:
            min_value = predict + 1
        elif number < predict:
            max_value = predict - 1
        count += 1
        predict = (min_value+max_value) // 2

    return(count)

# Проверяем
print('Алгоритм угадывает число в среднем за {} попыток'.format(game_score(game_core)))
