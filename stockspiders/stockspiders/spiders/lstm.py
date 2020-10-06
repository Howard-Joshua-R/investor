import os                                                  #used to get environment variable for alphaadvantage_apikey
import scrapy                                              #scrapy used to run scrapy spider
import logging                                             #logging used to supress junk logs from matplotlib
import pandas as pd                                        #allows you to store and manage data in tables
import numpy as np                                         #library for working with arrays (like tables ^)
import json                                                #we know what json is
import datetime as dt                                      #allows manipulating datetime objects
from sklearn.preprocessing import MinMaxScaler             #allows us to manipulate datasets to get a clean set, I think it's used to get data points between 0 and 1
import matplotlib.pyplot as plt                            #allows you to plot points and lines on a chart
from keras.models import Sequential                        #only useful if your model has 1 input and 1 output
from keras.layers import LSTM,Dropout,Dense                #lstm layer is long short-term memory, Dense is a layer that takes input from all previous cells both are used in the prediction model

class lstm(scrapy.Spider):          #this starts my class and accepts a generic scrapy spider
    name = "lstm"                   #required for scrapy to know spider name

    def start_requests(self):                       #function to start scrapy

        ticker = getattr(self, 'ticker', None)      #get arguement for ticker, if none proivded throw error
        if (ticker is None):
            raise ValueError('Please provide a ticker symbol!')

        logging.getLogger('matplotlib').setLevel(logging.WARNING) #suppress matplotlib logging
        logging.getLogger('tensorflow').setLevel(logging.WARNING)  # suppress logging

        apikey = os.getenv('alphavantage_apikey')                   #get api key from environment
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&apikey={1}&outputsize=full'.format(ticker, apikey) #url to get scrape data

        yield scrapy.Request(url, self.parse) #request url and call self.parse function

    def parse(self, response):                  #start parsing of url results

        print('Convert JSON to CSV')                #convert json data to a csv (wonder if I could do this with pandas?)
        data = json.loads(response.body)            #load the response
        data = data['Time Series (Daily)']          #narrow the scope to time series object
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open']) #create a pandas data frame/table
        for k, v in data.items():                                               #for each json item in data object
            date = dt.datetime.strptime(k, '%Y-%m-%d')                          #convert datetime to just a date format
            data_row = [date.date(), float(v['3. low']), float(v['2. high']),   #add the date and other data points to rows
                        float(v['4. close']), float(v['1. open'])]
            df.loc[-1, :] = data_row                                            #add data row to data frame
            df.index = df.index + 1                                             #increment the indexx of the data frame

        df = df.sort_values('Date')                                             #sort the dataframe by the 'date' column
        new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close']) #start new data set with just date and close price

        for i in range(0, len(df)):                     #foreach row in data frame, add to new dataset
            new_dataset["Date"][i] = df['Date'][i]      #date
            new_dataset["Close"][i] = df["Close"][i]    #close price

        new_dataset.index = new_dataset.Date            #set date column as index
        new_dataset.drop("Date", axis=1, inplace=True)  #remove date column since we now have it as the index

        final_dataset = new_dataset.values              #set final dataset
        totalDataPoints = len(final_dataset)            #get total days in dataset
        trainNum = round(totalDataPoints * .714)        #get 71% of data for training
        train_data = final_dataset[0:trainNum, :]       #set train data
        valid_data = final_dataset[trainNum:, :]        #set remaining valid data to test against

        scaler = MinMaxScaler(feature_range=(0, 1))         #create scaler
        scaled_data = scaler.fit_transform(final_dataset)   #convert all data points to decimals from 0.01 to 1.00

        x_train_data, y_train_data= [], []                  #create empty array for training data sets

        spread = 100                                            #I don't know what this does...***************
        for i in range (spread, len(train_data)):               #for start at spread go to length of train_data
            x_train_data.append(scaled_data[i-spread:i,0])      #add range to x _train_data
            y_train_data.append((scaled_data[i,0]))             #add range to y_train_data

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)              #convert to numpy array
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1],1)) #reshape the array but not sure what to?**********

        lstm_model = Sequential()                                                                           #the type of model to be used
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train_data.shape[1], 1)))     #not really sure what we're doing here
        lstm_model.add(LSTM(units=50))                                                                      #there are layers in the model
        #consider adding dropout here
        lstm_model.add(Dense(1))                                                                            #this is layer that connects neurons....

        lstm_model.compile(loss='mean_squared_error', optimizer='adam')                                 #build the model and train, i think
        lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)                   #forces the data to 'fit' in memeory, epochs is how many total times to train over the data. batch size is the length of a 'batch' in the data set

        inputs_data = new_dataset[len(new_dataset)-len(valid_data)-spread:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        X_test = []
        for i in range(spread, inputs_data.shape[0]):
            X_test.append(inputs_data[i-spread:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_closing_price = lstm_model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

        future = []
        currentStep = X_test[:,-1:,:]
        for i in range(1):
            currentStep = lstm_model.predict(currentStep)
            future.append(currentStep)

        train_data = new_dataset[:trainNum]
        valid_data = new_dataset[trainNum:]
        valid_data['Predictions'] = predicted_closing_price
        future['Future'] = future[0:]
        plt.plot(train_data["Close"], label='Training Data Close')
        plt.plot(valid_data['Close'], label='Actual Closing Price')
        plt.plot(valid_data['Predictions'], label='Predicted Closing Price')
        plt.plot(future['Future'], label="Future Predictions")
        plt.legend()
        plt.show()

        print('done')