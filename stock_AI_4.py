# This program has defined weights

import math
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
from db_handler import db_connect, db_close

plt.style.use('fivethirtyeight')
np.random.seed(4)


def data_graph(title, train, valid):
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    # plt.show()
    return plt


def lstm_network_model(x_train):
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=60, return_sequences=False))
    model.add(Dense(units=30))
    model.add(Dense(units=1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, batch_size=1, epochs=1) # epoch 5 is slower but more accurate


def model_prediction(model, scaler, x_test):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling
    return predictions


def calculate_rmse(predictions, y_test):
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    return rmse


def get_quote(ticker, start_date, end_date):
    quote = web.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)
    return quote


def get_scaler():
    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler


def get_scaled_data(scaler, data_set):
    scaled_data = scaler.fit_transform(data_set)
    return scaled_data


def get_training_data_len(data_set):
    # Get /Compute the number of rows to train the model on 80 percent of the historical data
    training_data_len = math.ceil(len(data_set) * .8)
    return training_data_len


def split_train_data(scaled_data, training_data_len):
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train


def create_test_data(scaled_data, training_data_len):
    # Create Test data set from index 1543 to 2003 the end
    test_data = scaled_data[training_data_len - 60:, :]
    return test_data


def split_test_data(data_set, training_data_len, test_data):
    # Create the x_test and y_test data sets
    x_test = []
    y_test = data_set[training_data_len:, :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)
    return x_test, y_test


def get_last_60_days(scaler, data):
    # Get teh last 60 day closing price
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    x_test = [last_60_days_scaled]
    # Convert the x_test data set to a numpy array
    x_test = np.array(x_test)
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


def get_current_quote(ticker):
    try:
        quote = web.DataReader(ticker, data_source='yahoo', start=datetime.date(datetime.now()), end=datetime.date(datetime.now()))
    except:
        quote = web.DataReader(ticker, data_source='yahoo', start=datetime.date(datetime.now() + timedelta(days=-1)), end=datetime.date(datetime.now() + timedelta(days=-1)))

    return quote['Close']


def main():
    ticker = input("Enter stock ticker: ").upper()
    start_date = input("Enter start training date in the following format (2012-01-01) : ")
    prediction_for = input("Prediction for today or tomorrow? : ").upper()

    if prediction_for == "TODAY":
        end_date = datetime.date(datetime.now()) + timedelta(days=-1)
    else:
        end_date = datetime.date(datetime.now())

    print(end_date)

    quote = get_quote(ticker, start_date, end_date)

    # Create a new dataframe with only the 'Close' column
    data = quote.filter(['Close'])

    # Converting the dataframe to a numpy array
    data_set = data.values

    training_data_len = get_training_data_len(data_set)

    scaler = get_scaler()
    scaled_data = get_scaled_data(scaler, data_set)

    x_train, y_train = split_train_data(scaled_data, training_data_len)

    # Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # get the model
    model = lstm_network_model(x_train)

    train_model(model, x_train, y_train)

    test_data = create_test_data(scaled_data, training_data_len)

    x_test, y_test = split_test_data(data_set, training_data_len, test_data)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model_prediction(model, scaler, x_test)

    margin_error = calculate_rmse(predictions, y_test)

    #### Plot/Create the data for the graph ####
    train = data[:training_data_len]

    # valid = data[training_data_len:].copy()
    valid = data[training_data_len:]

    # valid.loc[:,'Predictions'] = predictions
    valid['Predictions'] = predictions

    graph = data_graph(ticker, train, valid)

    print(valid)

    last_60_days = get_last_60_days(scaler, data)

    # Get the predicted scaled price
    pred_price = model_prediction(model, scaler, last_60_days)

    print("Price Prediction : ", pred_price, " with a margin error of ", margin_error)
    # print(pred_price)

    current_quote = get_current_quote(ticker)
    print("Current Price : ", current_quote)

    show_graph = input("Show Graph - Yes or No? : ").upper()

    if show_graph == "YES":
        graph.show()

    my_db = db_connect()
    db_close(my_db)


if __name__ == "__main__":
    main()

# how seeds affect the output?
