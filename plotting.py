import numpy as np
from matplotlib import pyplot as plt


def plot_metric(metric_train, metric_validation=None, xlabel='x', ylabel='y', title='Metric'):
    plt.plot(range(len(metric_train)), metric_train)

    if len(metric_validation) <= len(metric_train):
        domain_val = list(np.linspace(0, len(metric_train)-1, num=len(metric_validation)))
    else:
        domain_val = list(range(len(metric_validation)))

    legend = ['training']
    if metric_validation is not None:
        plt.plot(domain_val, metric_validation)
        legend.append('validation')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(legend, loc='upper left')
    plt.savefig(title + '.png')
    plt.show()
    plt.close()
