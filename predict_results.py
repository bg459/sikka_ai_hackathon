pth = 'data_file.csv'
import pandas as pd
df = pd.read_csv(pth)
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')


df['date'] = df['month'].astype(str) + '-' + df['year'].astype(str)

covid_months = ['3-2020', '4-2020', '5-2020', '6-2020']
df = df[~df['date'].isin(covid_months)]

# this is the set of arima models we decided on
orders = [(1, 0, 1), (1, 0, 2)]
# These id's end in november instead of december
issue_ids = [262, 252, 66]

ids = []
months = []
productions = []
## loop through the practices
for id in range(1, 285):
    temp = df[df['id'] == id]
    train_data = temp['production']

    models = []
    for o in orders:
        model = sm.tsa.statespace.SARIMAX(order = o, endog = train_data,\
                                          exog = None).fit(disp = -1)
        models.append(model)

    y_hats = []

    for model in models:
        if id in issue_ids:
            # want to get pred's for jan - apr
            y_hat = model.predict(0, len(train_data) + 6)[1:]
            y_hats.append(y_hat)
        else:
            y_hat = model.predict(0, len(train_data) + 5)[1:]
            y_hats.append(y_hat)

    y_hat = sum(y_hats)/len(y_hats)
    y_hat = list(y_hat)

    forecast = list(y_hat[-4:])

    productions = productions + forecast
    months = months + [1, 2, 3, 4]
    ids = ids + [id] * len(forecast) 

df = pd.DataFrame({'id': ids, 'month': months, 'production': productions})
df['year'] = 2021
print(df)
df.to_csv("result.csv", index = False)
