# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V2a.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/20 15:58:06 by vrabaib           #+#    #+#              #
#    Updated: 2019/07/25 17:35:05 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import random

# data = pd.read_csv(wine_data, delimiter = ';')

def rosenblatt_qlty_column(wine_data, good_threshold, bad_threshold):
    new_data = data.assign(Good=[1 if qlty >= good_threshold else 0 if qlty <= bad_threshold else -1 for qlty in data['quality']])
    
    threshold_filter = (new_data['quality'] >= good_threshold) | (new_data['quality'] <= bad_threshold)
    rosenblatt_quality = new_data[threshold_filter]
    return rosenblatt_quality

# updated_data = rosenblatt_qlty_column(data, 8, 3)
# print(updated_data[['pH', 'alcohol', 'quality', 'Good']])


class Perceptron():
    def __init__(self, data):
        self.data = data

    def heaviside_step_activation(self, activation):
        return 1 if activation >= 0 else 0

    def predict(self, params):
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += self.weights[i] * params[i]
        weighted_sum += self.bias
        return self.heaviside_step_activation(weighted_sum)
        
    def rosenblatt_learning(self):
        self.bias = random.random()
        self.weights = [random.random() for params in range(len(list(self.data)))]
        
        for itr in range(15000):
            errors = 0
            for x, y_true in zip(self.data.values, self.data['Good']):
                error = y_true - self.predict(x)
                if error != 0:
                    self.weights += error * x
                    self.bias += error
                    errors += 1
            if errors == 0:
                break

# test = Perceptron(updated_data[['pH', 'alcohol', 'Good']])
# test.rosenblatt_learning()