import numpy as np
import requests

from steps import WindowTimeSeries


def fetch_data(window_size):
    data_inputs_usd = load_currency("USD")
    data_inputs_eur = load_currency("EUR")

    data_inputs_usd = np.expand_dims(np.array(data_inputs_usd), axis=1)
    data_inputs_eur = np.expand_dims(np.array(data_inputs_eur), axis=1)
    data_inputs = np.concatenate((data_inputs_usd, data_inputs_eur), axis=1)

    return WindowTimeSeries(window_size=window_size).transform((data_inputs, None))


def load_currency(currency):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2017-03-03&currency={}".format(
            currency
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]

    return kept_values
