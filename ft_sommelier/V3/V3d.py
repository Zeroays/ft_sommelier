# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V3d.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/29 12:36:32 by vrabaib           #+#    #+#              #
#    Updated: 2019/08/01 12:51:13 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
sys.path.insert(1, '../V2')

from V2c import setup_plot, plot_performance
from V2d import average, mean_normalization
from V3c import infinity, vector_op, dot_product, Adaline
import pandas as pd
import random


data = pd.read_csv("../winequality-red.csv", delimiter = ';')
qlty_filtered_data = data[(data['quality'] >= 7) | (data['quality'] <= 4)]
new_data = qlty_filtered_data[['alcohol', 'pH', 'quality']]
two_param_data = new_data.assign(Good=[1 if qlty >= 7 else -1 for qlty in new_data['quality']])

x_axis_avg = average(two_param_data['alcohol'].values)
y_axis_avg = average(two_param_data['pH'].values)

two_param_data['alcohol'] = two_param_data['alcohol'].apply(lambda x : mean_normalization(x, two_param_data['alcohol'], x_axis_avg))
two_param_data['pH'] = two_param_data['pH'].apply(lambda y : mean_normalization(y, two_param_data['pH'], y_axis_avg))

#Gradient Descent - Batch : Learning Rate : 0.001
adaline = Adaline(two_param_data[['alcohol', 'pH']], 300, 0.001, True)
adaline.widrow_hoff_learning(two_param_data['Good'])

# print(two_param_data)
#print(adaline.performance)

#tmp = new_data.assign(Good=[1 if qlty >= 7 else -1 for qlty in new_data['quality']])
plot_performance(adaline.performance, two_param_data, 7, 4, 0, False)

