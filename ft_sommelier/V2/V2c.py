# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    V2c.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vrabaib <vrabaib@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/07/22 13:18:52 by vrabaib           #+#    #+#              #
#    Updated: 2019/08/01 12:44:31 by vrabaib          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from V2b import *
import matplotlib.pyplot as plt

updated_data = rosenblatt_qlty_column("../winequality-red.csv", 8, 3)
red_wine_perceptron = Perceptron(updated_data[['pH', 'alcohol']], 0.9, 0)
perceptron_performance = red_wine_perceptron.rosenblatt_learning(updated_data['Good'])

def setup_plot(plot, title, x_label, y_label):
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    return plot

def plot_performance(performance, wine_data, good_thresh, bad_thresh, epoch=-1, save_plot=False):
    """
    Plot the performance of our perceptron or adaline.
    This function will produce a two plot figure:
    1) Classification Errors vs Epochs
    2) Decision boundary for two factors
    """
    def extract(index):
        return [itr[index] for itr in performance]

    def y_val_decision_boundary(x, weight1, weight2, bias):
        return (-weight2 / weight1) * x + (-bias / weight1)

    x_axis, y_axis, compare = 'alcohol', 'pH', 'quality'
    epo_vs_error_title = 'Error as a function of epoch'
    e_vs_e_x_title, e_vs_e_y_title = 'epoch', 'classification errors'
    decision_boundary_title = 'Decision boundary on epoch'
    epochs, errors, weights, bias = extract(0), extract(1), extract(2), extract(3)
    epoch_choice = epoch if (epoch > 0 and epoch < len(performance)) else len(performance)

    fig, plot = plt.subplots(1, 2, figsize=(15, 5))
    epoch_vs_error = setup_plot(plot[0], epo_vs_error_title, e_vs_e_x_title, e_vs_e_y_title)
    x_param_vs_y_param = setup_plot(plot[1], decision_boundary_title + " : " + str(epoch_choice), x_axis, y_axis)
    
    epoch_buffer, errors_buffer = max(epochs) * 0.075, max(errors) * 0.075
    plot[0].set(xlim=(min(epochs) - epoch_buffer, max(epochs) + epoch_buffer), ylim=(min(errors) - errors_buffer, max(errors) + errors_buffer))
    plot[0].plot(epochs[:epoch_choice], errors[:epoch_choice], 'b--')

    b = bias[epoch_choice - 1]
    weight1 = weights[epoch_choice - 1][0]
    weight2 = weights[epoch_choice - 1][1]

    bad_wine = (wine_data[wine_data[compare] <= bad_thresh])[[x_axis, y_axis]]
    good_wine = (wine_data[wine_data[compare] >= good_thresh])[[x_axis, y_axis]]
    
    #Hard coded
    x_decision_boundary = [i for i in range(-1, 1)]
    y_decision_boundary = [y_val_decision_boundary(i, weight2, weight1, b) for i in range(-1, 1)]
    #End
    
    decision_boundary = plot[1].plot(x_decision_boundary, y_decision_boundary, 'b--')
    bad_wine_plot = plot[1].scatter(bad_wine[x_axis], bad_wine[y_axis], s=20, c='purple')
    good_wine_plot = plot[1].scatter(good_wine[x_axis], good_wine[y_axis], s=20, c='green')
    
    #Need to fix
    plot[1].fill_between(x_decision_boundary, y_decision_boundary, alpha=0.4)

    plot[1].legend(['decision boundary' , 'bad wines (<' + str(bad_thresh) + ' score)', 'good wines (>' + str(good_thresh) + ' score)'], loc='upper left', bbox_to_anchor=(1,1), prop={'size':6})
            
    if save_plot == True:
        plt.savefig("perceptron_plot_performance.png")

    plt.show()

#plot_performance(perceptron_performance, updated_data, 8, 3, 15000, True)