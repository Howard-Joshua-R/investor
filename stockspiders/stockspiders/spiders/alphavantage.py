import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import json
import os
import scrapy
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
import logging


def roundDown(x):
    return int(math.floor(x / 1000.0)) * 1000

class AlphaVantage(scrapy.Spider):
    name = "alphavantage"

    def start_requests(self):
        ticker = getattr(self, 'ticker', None)
        if (ticker is None):
            raise ValueError('Please provide a ticker symbol!')

        logging.getLogger('matplotlib').setLevel(logging.WARNING)

        apikey = os.getenv('alphavantage_apikey')
        print(apikey)
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&apikey={1}&outputsize=full'.format(ticker, apikey)

        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        ticker = getattr(self, 'ticker', None)

        #save file to text
        #with open(ticker + '.txt', 'wb') as f:
        #    f.write(response.body)

        #https://www.datacamp.com/community/tutorials/lstm-python-stock-market
        #convert json to csv
        print('Convert JSON to CSV')
        data = json.loads(response.body)
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
        for k,v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                        float(v['4. close']), float(v['1. open'])]
            df.loc[-1,:] = data_row
            df.index = df.index +1
        df = df.sort_values('Date')

        #save file to csv
        #df.to_csv(ticker + '.csv')

        #plot graph
        #plt.figure(figsize = (18, 9))
        #plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
        #plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500],rotation=45)
        #plt.xlabel('Date', fontsize=18)
        #plt.ylabel('Mid Price', fontsize=18)
        #plt.show()

        print('Build Data Sets')
        #todo: build closing price range
        high_prices = df.loc[:, 'High'].to_numpy()
        low_prices = df.loc[:, 'Low'].to_numpy()
        mid_prices = (high_prices+low_prices)/2.0

        totalDataPoints = len(mid_prices)
        trainNum = round(totalDataPoints*.714)
        testNum = round(totalDataPoints*.286)
        train_data = mid_prices[:trainNum]
        test_data = mid_prices[trainNum:]

        scaler = MinMaxScaler()
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)

        print('Train the Scaler with training data and smooth data')
        smoothing_window_size = 500
        roundVal = roundDown(trainNum)#todo write round down function
        for di in range(0, roundVal, smoothing_window_size):
            scaler.fit(train_data[di:di + smoothing_window_size, :])
            train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

        print('You normalize the last bit of remaining data')
        scaler.fit(train_data[di + smoothing_window_size:, :])
        train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

        print("reshape both train and test data")
        train_data = train_data.reshape(-1)

        print("normalize test data")
        test_data = scaler.transform(test_data).reshape(-1)

        # Now perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
        EMA = 0.0
        gamma = 0.1
        for ti in range(trainNum): #changed 11000 to 800
            EMA = gamma*train_data[ti] + (1-gamma)*EMA
            train_data[ti] = EMA

        # Used for visualization and test purposes
        all_mid_data = np.concatenate([train_data,test_data], axis=0)

        window_size = 100
        N = train_data.size
        std_avg_predictions = []
        std_avg_x = []
        mse_errors = []

        for pred_idx in range(window_size,N):
            if pred_idx >= N:
                date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
            else:
                date = df.loc[pred_idx, 'Date']

            std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
            mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
            std_avg_x.append(date)

        print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

        plt.figure(figsize=(18, 9))
        plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
        plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
        #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Mid Price')
        plt.legend(fontsize=18)
        plt.show()

        window_size = 100
        N = train_data.size

        run_avg_predictions = []
        run_avg_x = []

        mse_errors = []

        running_mean = 0.0
        run_avg_predictions.append(running_mean)

        decay = 0.5

        for pred_idx in range(1, N):
            running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
            run_avg_predictions.append(running_mean)
            mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
            run_avg_x.append(date)

        print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))

        plt.figure(figsize=(18, 9))
        plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
        plt.plot(range(0, N), run_avg_predictions, color='orange', label='Prediction')
        # plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Mid Price')
        plt.legend(fontsize=18)
        plt.show()

        print('done')