# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V1a.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/19 14:50:41 by vrabaib           #+#    #+#              #
#    Updated: 2019/07/24 13:26:53 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../winequality-red.csv", delimiter = ';')

def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
    params = list(data)
    param_amt = len(params) - 1

    plt.rc('xtick', labelsize=2)
    plt.rc('ytick', labelsize=2)
    
    fig, plot = plt.subplots(param_amt, param_amt, figsize=(20, 12))
    for i in range(param_amt):
        for j in range(param_amt):
            if i == j:
                plot[i, i].text(0.5, 0.5, params[i], fontsize=6, ha='center', va='center')
            else:
                colors = ['green' if qlty >= good_threshold else 'purple' for qlty in data['quality']]
                size = [0.1 if qlty >= good_threshold else 0.1 if qlty <= bad_threshold else 0 for qlty in data['quality']]
                plot[i, j].scatter(data[params[i]], data[params[j]], s=size, c=colors)

    if save_plot:
        plt.savefig('ft_sommelier.png')

    plt.show()


plot_scatter_matrix(data, 7, 4, True)