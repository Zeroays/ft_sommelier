# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V2b.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/21 18:39:40 by vrabaib           #+#    #+#              #
#    Updated: 2019/07/25 17:35:19 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import random

def rosenblatt_qlty_column(wine_data, good_threshold, bad_threshold):
    data = pd.read_csv(wine_data, delimiter = ';')
    new_data = data.assign(Good=[1 if qlty >= good_threshold else 0 if qlty <= bad_threshold else -1 for qlty in data['quality']])
    
    threshold_filter = (new_data['quality'] >= good_threshold) | (new_data['quality'] <= bad_threshold)
    rosenblatt_quality = new_data[threshold_filter]
    return rosenblatt_quality

updated_data = rosenblatt_qlty_column("../winequality-red.csv", 8, 3)
print(updated_data[['pH', 'alcohol', 'quality', 'Good']])

def infinity():
    index = 0
    while 1:
        yield index
        index += 1

class Perceptron():
    def __init__(self, data, learning_rate, epochs):
        self.data = data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.performance = []

    def heaviside_step_activation(self, activation):
        return 1 if activation >= 0 else 0

    def predict(self, params):
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += self.weights[i] * params[i]
        weighted_sum += self.bias
        return self.heaviside_step_activation(weighted_sum)
        
    def rosenblatt_learning(self, y_cmp):
        random.seed(9000)
        self.bias = random.random()
        self.weights = [random.random() for params in range(len(list(self.data)))]
        
        for itr in infinity() if self.epochs == 0 else range(self.epochs):
            errors = 0
            for x, y_true in zip(self.data.values, y_cmp):
                error = y_true - self.predict(x)
                print(x, y_true)
                if error != 0:
                    self.weights += error * x * self.learning_rate
                    self.bias += error * self.learning_rate
                    errors += 1
            self.performance.append((itr, errors, list(self.weights), self.bias))
            if errors == 0:
                break
        return self.performance

test = Perceptron(updated_data[['pH', 'alcohol']], 0.9, 0)
print(test.rosenblatt_learning(updated_data['Good']))