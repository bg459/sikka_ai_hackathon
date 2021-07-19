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

import json

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

df['date'] = df['month'].astype(str) + '-' + df['year'].astype(str)
def rolling_validate(id, order, exog = False):
    """
    This method does the rolling CV method on a given id practice. Starts rolling CV on the third instance of Jan, or 1-2020, whichever is earlier.
    At each step, trains on all prev. data and predicts on Jan-Apr. Computes net MAP score as outlined in competition
    rules.
    """
    # total_score = 0
    total_scores_list = [0 for i in range(len(orders))]
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
            for ol in order:
                curr_models = []
                for o in ol: 
                    model = sm.tsa.statespace.SARIMAX(order = o, endog = train_X,\
                                            exog = None).fit(disp = -1)
                    curr_models.append(model)
                models.append(curr_models)
        except:
            return -2

        y_hats = []
            
        for model_list in models:
            y_hat_curr = []
            
            for model in model_list:
                y_hat = list(model.predict(0, len(train_X) + 5)[1:])
                y_hat_curr.append(y_hat)
            
            # average yhat values
            y_hats.append([sum(x)/len(x) for x in zip(*y_hat_curr)])
        
        scores = []
        if len(months) > 1:
            # calculate scores for each model
            for ind in range(len(y_hats)):
                y_hat = y_hats[ind]

                forecast = np.array(y_hat[-4:])
                forecast = forecast[[x-1 for x in months]]
            
                mape_arima = mean_absolute_percentage_error(y, forecast)
            
                score_arima = 2 if mape_arima <= 5 else 1 if mape_arima <= 10 else -1 if mape_arima <= 15 else -2
                scores.append(score_arima)
            
            # combine scores with prev years to get total
            total_scores_list = [x + y for x, y in zip(scores, total_scores_list)]
            i+=1

        start_year += 1

    # find order with max score
    max_score  = -1000
    order_curr = 0
    for ind in range(len(total_scores_list)):
        if max_score < total_scores_list[ind]:
            max_score = total_scores_list[ind]
            order_curr = orders[ind]
    
    if i == 0:
        return -2, order_curr

    return max_score/i, order_curr

## EXAMPLE RUN: Can change contents of "orders" to test different model(s).

# orders = [(1, 0, 1), (0, 1, 2), (0, 1, 1), (1, 1, 1), (1, 0, 2), (1, 0, 3)]
orders_init = [(1, 0, 1), (1, 0, 2), (0, 1, 2), (0, 1, 1), (1, 1, 1), (1, 1, 2), (3, 1, 1), (2, 1, 1), (4, 1, 0), (3, 1, 0)]

# make each order a list of orders for model ensembling 
orders = []
for a in orders_init:
    for b in orders_init:
        orders.append([a, b])

id2order = {}
s = 0
for id in tqdm(range(1, 285)):
    curr, order = rolling_validate(id, orders)
    id2order[id] = order
    print(id, curr, order)
    s += curr

print(s)
with open('arima_orders.json', 'w') as f:
    json.dump(id2order, f)

