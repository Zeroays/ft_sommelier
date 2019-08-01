# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V3b.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/24 13:29:59 by vrabaib           #+#    #+#              #
#    Updated: 2019/07/27 21:32:45 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import random

data = pd.read_csv("../winequality-red.csv", delimiter = ';')
qlty_filtered_data = data[(data['quality'] >= 7) | (data['quality'] <= 4)]
two_param_data = qlty_filtered_data[['alcohol', 'pH', 'quality']]

def vector_op(v1, v2, op):
    ops = {
        "+" : lambda x, y : x + y,
        "-" : lambda x, y : x - y,
        "*" : lambda x, y : x * y,
        "/" : lambda x, y : x / y
    }
    return [ops[op](num1,num2) for num1, num2 in zip(v1, v2)]

def dot_product(matrix, vector):
    sums = []
    for rows in matrix:
        weighted_sum = 0
        for i in range(len(vector)):
            weighted_sum += rows[i] * vectors[i]
        sums.append(weighted_sum)
    return sums
        
class Adaline():
    def __init__(self, data):
        self.data = data

    def linear_activation(self, activation):
        return activation

    def net_input(self, params):
        bias_vector = [self.bias] * len(params)
        return vector_op(dot_product(params, self.weights), bias_vector, "+")

    def widrow_hoff_learning(self, cmp_data):
        random.seed(9000)
        self.costs = []
        self.bias = random.random()
        self.weights = [random.random() for params in range(len(list(self.data)) - 1)]

        for i in range(100):
            output = [self.linear_activation(samples) for samples in self.net_input(self.data.values)]
            errors = vector_subtract(cmp_data.values, output)
            self.weights += dot_product(self.data.values, errors)
            self.bias += sum(errors)
            cost = sum([error**2 for error in errors]) / 2.0
            self.costs.append(cost)

    def predict():
        return [1 if result >= 0 else -1 for result in self.net_input(self.data.values)]
