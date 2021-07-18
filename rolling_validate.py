"""
This file contains the training and validation code that we use to choose our best model.
The rolling_validate() function uses a rolling window of train/test splits across time series
data and computes the overall score based on MAPE according to competition rules.

We call this function iteratively to determine the best combination of models. This function
supports both individual and ensembling model ideas.
"""


pth = 'data_file.csv'
import pandas as pd
df = pd.read_csv(pth)
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


df['date'] = df['month'].astype(str) + '-' + df['year'].astype(str)
def rolling_validate(id, order, exog = False):
    """
    This method does the rolling CV method on a given id practice. Starts rolling CV on the third instance of Jan, or 1-2020, whichever is earlier.
    At each step, trains on all prev. data and predicts on Jan-Apr. Computes net MAP score as outlined in competition
    rules.
    """
    total_score = 0
    temp = df[df['id'] == id]
    januaries = temp[temp['month'] == 1]
    if len (januaries) <= 3:
        start_date = '1-2020'
    else:
        start_date = list(januaries['date'])[2]

    def year_from_date(date):
        return int(date.rsplit('-')[1])

    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    start_year = year_from_date(start_date)
    i = 0
    while start_year <= 2020:
        train_data = temp[temp['year'] < start_year]

        test_data = temp[temp['year'] == start_year]

        if start_year == 2020:
            test_data = test_data[test_data['month'].isin([1, 2])]
        else:
            test_data = test_data[test_data['month'].isin([1, 2, 3, 4])]


        y = np.array(test_data['production'])
        months = list(test_data['month'])
        ### This needs to be specified per model. prototyping ###
        ## Sometimes test data is not clean. For example, may only contain ground truth for 1, 2, 4.

        train_X = train_data['production']

        # get some list of models based on order
        models = []
        
        try:
            for o in order:
                model = sm.tsa.statespace.SARIMAX(order = o, endog = train_X,\
                                          exog = None).fit(disp = -1)
                models.append(model)
        except:
            return -2

        y_hats = []
        for model in models:
            y_hat = model.predict(0, len(train_X) + 5)[1:]
            y_hats.append(y_hat)

        y_hat = sum(y_hats)/len(y_hats)
        y_hat = list(y_hat)

        # Assert: y_hat same shape as y, some list

        forecast = np.array(y_hat[-4:])
        forecast = forecast[[x-1 for x in months]]
        mape = mean_absolute_percentage_error(y, forecast)
        score = 2 if mape <= 5 else 1 if mape <= 10 else -1 if mape <= 15 else -2
        total_score += score

        i+=1
        start_year += 1

    return total_score/i

import warnings
warnings.filterwarnings('ignore')

s = 0


## EXAMPLE RUN: Can change contents of "orders" to test different model(s).

orders = [(1, 0, 1), (0, 1, 2), (0, 1, 1), (1, 1, 1), (1, 0, 2), (1, 0, 3)]
for id in range(1, 285):
    s+=rolling_validate(id, orders)

print(s)
