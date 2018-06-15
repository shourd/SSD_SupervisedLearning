import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_style("darkgrid")


def plot_acc_data(plot_name, dir_name, figsize):

    filename = 'acc_{}.csv'.format(plot_name)
    plot_title = ''
    output_file = 'acc_{}.pdf'.format(plot_name)

    filepath = dir_name + filename
    acc_data = np.genfromtxt(filepath, delimiter=',')
    acc_data = acc_data.transpose()

    # x axis
    start = 1
    stop = len(acc_data)
    step = 1
    epochs = np.arange(start, stop+step, step)

    """ INPUT LEGEND HERE as labels """
    if plot_name == 'classes':
        labels = ['2 classes', '4 classes', '6 classes', '8 classes', '10 classes', '12 classes']

    if plot_name == 'dimensions':
        labels = ['120x120 px', '96x96 px', '64x64 px', '32x32 px', '16x16 px']

    if plot_name == 'rotations':
        labels = ['0 deg', '1 deg', '2 deg', '3 deg', '4 deg', '5 deg']

    if plot_name == 'architectures':
        labels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

    if plot_name == 'samples':
        samples = [150, 300, 500, 1000, 1500, 2000, 3000, 'all']
        labels = []
        for number in samples:
            labels.append(str(number) + ' samples')

    if plot_name == 'randomness':
        labels = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = [''.join([str(label), '%']) for label in labels]

    plt.figure(figsize=figsize)
    fig = plt.plot(epochs, acc_data)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.legend(fig, labels, loc='lower right')
    if plot_name == 'randomness':
        plt.legend(fig, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(np.arange(0, stop+1, 2))
    plt.xlim(0, len(acc_data))
    plt.ylim(0, 1)
    plt.savefig(dir_name + output_file, format='pdf', dpi=1000, bbox_inches='tight')
    print('Figure saved to {}'.format(dir_name + output_file))
    # plt.show()


def plot_val_data(plot_name, dir_name, figsize):
    filename = 'val_acc_list.csv'
    plot_title = ''
    output_file = 'val_acc_{}.pdf'.format(plot_name)

    filepath = dir_name + filename
    with open(filepath, 'r') as file:
        content = file.readlines()

    # x-axis
    if plot_name == 'classes':
        x_data = [2, 4, 6, 8, 10, 12]
        xlabel = 'Classes'
        y_data = content[3]

    elif plot_name == 'dimensions':
        x_data = [120, 96, 64, 32, 16]
        xlabel = 'Input dimensions (px x px)'
        y_data = content[0]

    elif plot_name == 'rotations':
        x_data = [0, 1, 2, 3, 4, 5]
        xlabel = 'Maximum rotation during augmentation'
        y_data = content[2]

    elif plot_name == 'architectures':
        x_data = [1, 2, 3, 4, 5, 6, 7]
        xlabel = 'Architecture number'
        y_data = content[1]

    elif plot_name == 'samples':
        x_data = [150, 300, 500, 1000, 1500, 2000, 3000, 6000]
        xlabel = 'Input samples'
        y_data = content[4]

    elif plot_name == 'randomness':
        x_data = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        xlabel = '% randomized resolutions'
        y_data = content[5]

    y_data = y_data.split(',')
    last_item = len(y_data) - 1
    y_data[last_item] = y_data[last_item].replace("\n", "")
    y_data = [float(i) for i in y_data]

    plt.figure(figsize=figsize)
    fig = plt.scatter(x_data, y_data)
    plt.xlabel(xlabel)
    plt.ylabel('Validation accuracy')
    plt.title(plot_title)
    plt.ylim(0, 1)
    plt.savefig(dir_name + output_file, format='pdf', dpi=1000, bbox_inches='tight')
    print('Validation figure saved to {}'.format(dir_name + output_file))


def plot_valtest_data(plot_name, dir_name, figsize):
    filename_val = 'val_acc_list.csv'
    filename_test = 'test_acc_{}.csv'.format(plot_name)
    plot_title = ''
    output_file = 'comb_acc_{}.pdf'.format(plot_name)
    markersize = 12
    alpha = 1  # transparency of the markers

    filepath = dir_name + filename_val
    with open(filepath, 'r') as file:
        content_val = file.readlines()

    filepath = dir_name + filename_test
    with open(filepath, 'r') as file:
        content_test = file.readlines()

    # x-axis
    if plot_name == 'classes':
        x_data = [2, 4, 6, 8, 10, 12]
        xlabel = 'Classes'
        y_data_val = content_val[3]

    elif plot_name == 'dimensions':
        x_data = [120, 96, 64, 32, 16]
        xlabel = 'Input dimensions (px x px)'
        y_data_val = content_val[0]

    elif plot_name == 'rotations':
        x_data = [0, 1, 2, 3, 4, 5]
        xlabel = 'Maximum rotation during augmentation'
        y_data_val = content_val[2]

    elif plot_name == 'architectures':
        x_data = [1, 2, 3, 4, 5, 6, 7]
        xlabel = 'Architecture number'
        y_data_val = content_val[1]

    elif plot_name == 'samples':
        x_data = [150, 300, 500, 1000, 1500, 2000, 3000, 6000]
        xlabel = 'Input samples'
        y_data_val = content_val[4]

    elif plot_name == 'randomness':
        x_data = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        xlabel = '% randomized resolutions'
        y_data_val = content_val[5]

    y_data_test = content_test[0]

    y_data_val = y_data_val.split(',')
    last_item = len(y_data_val) - 1
    y_data_val[last_item] = y_data_val[last_item].replace("\n", "") # remove \n
    y_data_val = list(filter(None, y_data_val))  # remove empty items
    y_data_val = [float(i) for i in y_data_val]

    y_data_test = y_data_test.split(',')
    last_item = len(y_data_test) - 1
    y_data_test[last_item] = y_data_test[last_item].replace("\n", "")
    y_data_test = [float(i) for i in y_data_test]

    plt.figure(figsize=figsize)
    val_set = plt.scatter(x_data, y_data_val, s=markersize, alpha=alpha)
    test_set = plt.scatter(x_data, y_data_test, s=markersize, alpha=alpha)
    plt.clabel(val_set, inline=1, fontsize=10)
    plt.show()
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.ylim(0, 1)
    labels = ['Validation set', 'Test set']
    plt.legend((val_set, test_set), labels, loc='lower right')
    plt.savefig(dir_name + output_file, format='pdf', dpi=1000, bbox_inches='tight')
    print('Combination figure saved to {}'.format(dir_name + output_file))


if __name__ == "__main__":
    # plot_names = ['dimensions', 'architectures', 'rotations', 'classes', 'samples', 'randomness']
    plot_names = ['randomness']
    dir_name = 'output6juni/'
    figsize = (5, 4)

    for plot_name in plot_names:
        plot_acc_data(plot_name, dir_name, figsize)
        # plot_val_data(plot_name, dir_name, figsize)
        # plot_valtest_data(plot_name, dir_name, figsize)


# ARCHITECTURES 1
# labels = ['64 CONV, 2x2 POOL',
#           '32 CONV, 64 CONV, 2x2 POOL',
#           '32 CONV, 32 CONV, 64 CONV, 2x2 POOL',
#           '32 CONV, 32 CONV, 2x2 POOL, 64 CONV, 2x2 POOL',
#           '32 CONV, 32 CONV, 2x2 POOL, 32 CONV, 64 CONV, 2x2 POOL',
#           '32 CONV, 32 CONV, 2x2 POOL, 32 CONV, 2x3 POOL, 64 CONV, 2x2 POOL']


# ARCHITECTURES 2
# labels = ['32 CONV, 2x2 POOL',
#           '64 CONV, 2x2 POOL',
#           '32 CONV, 4x4 POOL',
#           '32 CONV, 2x2 POOL + 64 CONV + 2x2 POOL',
#           '32 CONV, 2x2 POOL + 64 CONV + 4x4 POOL']