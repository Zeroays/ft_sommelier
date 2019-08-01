# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V3c.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/27 02:32:59 by vrabaib           #+#    #+#              #
#    Updated: 2019/08/01 12:23:02 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import random
import math

data = pd.read_csv("../winequality-red.csv", delimiter = ';')
qlty_filtered_data = data[(data['quality'] >= 7) | (data['quality'] <= 4)]
two_param_data = qlty_filtered_data[['alcohol', 'pH', 'quality']]

def infinity():
    index = 0
    while 1:
        yield index
        index += 1

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
    for i in range(len(matrix)):
        weighted_sum = 0
        for j in range(len(vector)):
            weighted_sum += matrix[i][j] * vector[j]
        sums.append(weighted_sum)
    return sums

def transpose(matrix, row_size):
    t = []
    for i in range(row_size):
        t.append([row[i] for row in matrix])
    return t

class Adaline():
    def __init__(self, data, epochs, learning_rate, batch=False):
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch = batch

    def linear_activation(self, activation):
        return activation

    def net_input(self, params):
        bias_vector = [self.bias] * len(params)
        return vector_op(dot_product(params, self.weights), bias_vector, "+")

    def calc_errors(self, cmp_data, output):
        errors = 0
        for i in range(len(cmp_data)):
            if cmp_data[i] != output[i]:
                errors += 1
        return errors

    def online_learning(self, cmp_data, epoch):
        costs = []
        print(self.weights)
        for x, y_true in zip(self.data.values, cmp_data):
            output = self.net_input([x])
            error = y_true - output[0]
            self.weights = vector_op(self.weights, vector_op([self.learning_rate * error] * len(self.weights), x, "*"), "+")
            self.bias += self.learning_rate * error
            cost = (error**2) / 2.0
            costs.append(cost)
        avg_cost = sum(costs) / len(cmp_data)
        self.costs.append(avg_cost)
        performance = (epoch, self.calc_errors(cmp_data.values, self.predict()), self.weights, self.bias)
        self.performance.append(performance)

    def batch_learning(self, cmp_data, epoch):
        output = [self.linear_activation(samples) for samples in self.net_input(self.data.values)]
        errors = vector_op(cmp_data.values, output, "-")
        self.weights = vector_op(self.weights, vector_op([self.learning_rate] * 2, dot_product(transpose(self.data.values, 2), errors), "*"), "+")
        self.bias += self.learning_rate * sum(errors)
        cost = sum([error**2 for error in errors]) / 2.0
        self.costs.append(cost)
        performance = (epoch, self.calc_errors(cmp_data.values, self.predict()), self.weights, self.bias)
        self.performance.append(performance)
        

    def widrow_hoff_learning(self, cmp_data):
        random.seed(9000)
        self.performance = []
        self.costs = []
        self.bias = random.random()
        #self.bias = random.random()
        #self.weights = 
        self.weights = [random.random() for params in range(len(list(self.data)))]
        for i in infinity() if self.epochs == 0 else range(self.epochs):
            if self.batch == True:
                self.batch_learning(cmp_data, i)
                
            else:
                self.online_learning(cmp_data, i)
            # if (i > 2 and (abs(self.costs[i - 1]) - abs(self.costs[i]) <= 0.001)):
            #     break

    def predict(self):
        return [1 if result >= 0 else -1 for result in self.net_input(self.data.values)]
