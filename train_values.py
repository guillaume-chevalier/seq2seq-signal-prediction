def plot_metrics(metric_name, train_values, metric_validation, exercice_number):
    print('last mse train: {}'.format(train_values[-1]))
    print('best mse train: {}'.format(min(mse_train)))

    mse_validation = 0
    print('last mse validation: {}'.format(mse_validation[-1]))
    print('best mse validation: {}'.format(min(mse_validation)))

    plot_metric(
        mse_train,
        mse_validation,
        xlabel='epoch',
        ylabel='mse',
        title='Exercice {} Model Mean Squared Error'.format(exercice_number)
    )