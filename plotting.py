import numpy as np
from matplotlib import pyplot as plt


def plot_predictions(data_inputs, expected_outputs, predicted_outputs, save=False):
    plt.figure(figsize=(12, 3))

    for output_dim_index in range(predicted_outputs.shape[-1]):
        past = data_inputs[:, output_dim_index]
        expected = expected_outputs[:, output_dim_index]
        pred = predicted_outputs[:, output_dim_index]

        label1 = "Seen (past) values" if output_dim_index == 0 else "_nolegend_"
        label2 = "True future values" if output_dim_index == 0 else "_nolegend_"
        label3 = "Predictions" if output_dim_index == 0 else "_nolegend_"

        plt.plot(range(past.shape[0]), past, "o--b", label=label1)
        plt.plot(range(past.shape[0], expected.shape[0] + past.shape[0]), expected, "x--b", label=label2)
        plt.plot(range(past.shape[0], pred.shape[0] + past.shape[0]), pred, "o--y", label=label3)

    plt.legend(loc='best')
    title = "Exercice Predictions v.s. true values"
    plt.title(title)

    if save:
        plt.savefig(title + '.png')

    plt.show()


def plot_metrics(pipeline, exercice_number):
    mse_train = pipeline.get_step_by_name('epoch_metrics').get_metric_train('mse')
    print('last mse train: {}'.format(mse_train[-1]))
    print('best mse train: {}'.format(min(mse_train)))

    mse_validation = pipeline.get_step_by_name('epoch_metrics').get_metric_validation('mse')
    print('last mse validation: {}'.format(mse_validation[-1]))
    print('best mse validation: {}'.format(min(mse_validation)))

    plot_metric(
        mse_train,
        mse_validation,
        xlabel='epoch',
        ylabel='mse',
        title='Exercice {} Model Mean Squared Error'.format(exercice_number)
    )


def plot_metric(metric_train, metric_validation=None, xlabel='x', ylabel='y', title='Metric', save=False):
    plt.plot(range(len(metric_train)), metric_train)

    if len(metric_validation) <= len(metric_train):
        domain_val = list(np.linspace(0, len(metric_train) - 1, num=len(metric_validation)))
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
    if save:
        plt.savefig(title + '.png')
    plt.show()
    plt.close()
