# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V2d.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/23 19:07:09 by vrabaib           #+#    #+#              #
#    Updated: 2019/07/24 13:26:38 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from V2b import rosenblatt_qlty_column, Perceptron
from V2c import setup_plot, plot_performance

def average(data):
    return sum(data) / len(data)

def mean_normalization(num, data, average):
    return (num - average) / (max(data.values) - min(data.values))

def with_feature_scaling(scaling=False):
    x_axis, y_axis = 'alcohol', 'pH'

    updated_data = rosenblatt_qlty_column("../winequality-red.csv", 8, 3)
    graph_data = updated_data[[x_axis, y_axis]]

    x_axis_avg = average(updated_data[x_axis].values)
    y_axis_avg = average(updated_data[y_axis].values)

    if scaling == True:
        graph_data['alcohol'] = graph_data['alcohol'].apply(lambda x : mean_normalization(x, graph_data['alcohol'], x_axis_avg))
        graph_data['pH'] = graph_data['pH'].apply(lambda y : mean_normalization(y, graph_data['pH'], y_axis_avg))

    red_wine_perceptron = Perceptron(graph_data[['pH', 'alcohol']], 0.3, 0)
    perceptron_performance = red_wine_perceptron.rosenblatt_learning(updated_data['Good'])
    plot_performance(perceptron_performance, updated_data, 8, 3, 300, True)

with_feature_scaling(True)