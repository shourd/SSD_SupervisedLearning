import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
sns.set_style("darkgrid")


def plot_acc_data(plot_name, dir_name):

    filename = 'acc_{}.csv'.format(plot_name)
    figsize = (5, 4)
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

    plt.figure(figsize=figsize)
    fig = plt.plot(epochs, acc_data)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.legend(fig, labels, loc='lower right')
    plt.xticks(np.arange(0, stop+1, 2))
    plt.xlim(0, len(acc_data))
    plt.savefig(dir_name + output_file, format='pdf', dpi=1000, bbox_inches='tight')
    print('Figure saved to {}'.format(dir_name + output_file))
    # plt.show()


def plot_val_data(plot_name, dir_name):
    filename = 'val_acc_list.csv'
    figsize = (5, 4)
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

    if plot_name == 'dimensions':
        x_data = [16, 32, 64, 96, 120]
        xlabel = 'Input dimensions (px x px)'
        y_data = content[0]

    if plot_name == 'rotations':
        x_data = [0, 1, 2, 3, 4, 5]
        xlabel = 'Maximum rotation during augmentation'
        y_data = content[2]

    if plot_name == 'architectures':
        x_data = [1, 2, 3, 4, 5, 6, 7]
        xlabel = 'Architecture number'
        y_data = content[1]

    if plot_name == 'samples':
        x_data = [150, 300, 500, 1000, 1500, 2000, 3000, 6000]
        xlabel = 'Input samples'
        y_data = content[4]

    y_data = y_data.split(',')
    last_item = len(y_data) - 1
    y_data[last_item] = y_data[last_item].replace("\n", "")
    y_data = [float(i) for i in y_data]

    plt.figure(figsize=figsize)
    fig = plt.scatter(x_data, y_data)
    plt.xlabel(xlabel)
    plt.ylabel('Validation accuracy')
    plt.title(plot_title)
    # plt.legend(fig, labels, loc='lower right')
    # plt.xticks(np.arange(0, stop+1, 2))
    # plt.xlim(0, len(my_data))
    plt.savefig(dir_name + output_file, format='pdf', dpi=1000, bbox_inches='tight')
    print('Validation figure saved to {}'.format(dir_name + output_file))

if __name__ == "__main__":
    plot_names = ['dimensions', 'architectures', 'rotations', 'classes', 'samples']
    dir_name = 'output6juni/'

    for plot_name in plot_names:
        plot_acc_data(plot_name, dir_name)
        plot_val_data(plot_name, dir_name)


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