# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:01:26 2019

@author: Amit & Sanchi
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_2_multi(data, tickers: list, ground_features: int = 5, new_features: int = 4):
    n, d = data.shape
    new_d = int(d/ground_features)
    new_data = np.zeros((n, new_d * new_features))
    open_prices = np.zeros((n, new_d))
    for i in range(new_d):
        new_data[:, new_features * i] = \
            data.iloc[:, ground_features * i]/data.iloc[:, ground_features * i + 3] - 1  # Returns
        new_data[:, new_features * i + 1] = \
            data.iloc[:, ground_features * i + 1] - data.iloc[:, ground_features * i + 2]  # Spread
        new_data[:, new_features * i + 2] = \
            (data.iloc[:, ground_features * i + 4] - np.min(data.iloc[:, ground_features * i + 4]))/ \
            (np.max(data.iloc[:, ground_features * i + 4]) - np.min(data.iloc[:, ground_features * i + 4]))  # Volume
        new_data[:, new_features * i + 3] = \
            (data.iloc[:, ground_features * i + 3] - np.min(data.iloc[:, ground_features * i + 3]))/ \
            (np.max(data.iloc[:, ground_features * i + 3]) - np.min(data.iloc[:, ground_features * i + 3]))  # open prize
        open_prices[:, i] = data.iloc[:, ground_features * i + 3]
    header_data = []
    header_open = []
    for ticker in tickers:
        header_data.append(ticker + '_returns')
        header_data.append(ticker + '_spread')
        header_data.append(ticker + '_volume')  # Normalized
        header_data.append(ticker + '_normalized_open')
        header_open.append(ticker + '_open')
    return pd.DataFrame(new_data, columns=header_data), pd.DataFrame(open_prices, columns=header_open)


def minutizer(data, split: int = 5, ground_features: int = 5):
    n, d = data.shape
    new_data = pd.DataFrame(np.zeros((int(n/split) - 1, d)), columns=list(data))
    for i in range(int(n/split) - 1):
        for j in range(int(d/ground_features)):
            # Close
            new_data.iloc[i, j * ground_features] = data.iloc[split * (i + 1), j * ground_features]
            # High
            new_data.iloc[i, j * ground_features + 1] = max([data.iloc[split*i+k, j * ground_features + 1]
                                                             for k in range(split)])
            # Low
            new_data.iloc[i, j * ground_features + 2] = min([data.iloc[split * i + k, j * ground_features + 2]
                                                             for k in range(split)])
            # Open
            new_data.iloc[i, j * ground_features + 3] = data.iloc[split*i, j * ground_features + 3]
            # Volume
            new_data.iloc[i, j * ground_features + 4] = np.sum(data.iloc[i*split:(i+1)*split, j * ground_features + 4])
    return new_data


def combine_ts(tickers: list):
    stock0 = tickers[0]
    path = 'C:/Users/Amit/src/data/project/'+stock0+'.csv'
    data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    renamer = {'close': stock0+'_close', 'high': stock0+'_high', 'low': stock0+'_low',
               'open': stock0+'_open', 'volume': stock0+'_volume', }
    data = data.rename(columns=renamer)
    tickers.remove(tickers[0])
    for str in tickers:
        path = 'C:/Users/Amit/src/data/project/'+str+'.csv'
        new_data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        renamer = {'close': str+'_close', 'high': str+'_high', 'low': str+'_low',
                   'open': str+'_open', 'volume': str+'_volume', }
        new_data = new_data.rename(columns=renamer)
        data = pd.concat([data, new_data], axis=1, sort=True)

    tickers.insert(0, stock0)
    return data.interpolate()[1:data.shape[0]]


def customized_loss(y_pred, y_true):
    num = K.sum(K.square(y_pred - y_true), axis=-1)
    y_true_sign = y_true > 0
    y_pred_sign = y_pred > 0
    logicals = K.equal(y_true_sign, y_pred_sign)
    logicals_0_1 = K.cast(logicals, 'float32')
    den = K.sum(logicals_0_1, axis=-1)
    return num/(1 + den)

## input parameters
def run_LSTM(stocks: list):
   
    lookback: int = 24
    epochs: int = 50
    batch_size: int = 96
    learning_rate: float = 0.0002
    dropout_rate: float = 0.1
    ground_features: int = 4
    
    # percentile: int = 10
    # import data
    data = combine_ts(stocks)
    data = minutizer(data, split=5)
    data, _ = preprocess_2_multi(data, stocks)
    # transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
    
    X = np.zeros((n - lookback, lookback, d))
    Y = np.zeros((n - lookback, int(d/ground_features)))
    for i in range(X.shape[0]):
        for j in range(d):
            X[i, :, j] = data.iloc[i:(i+lookback), j]
            if j < int(d/ground_features):
                Y[i, j] = data.iloc[lookback + i, j * ground_features]
    
    X_train = X[0: int(n * train_val_test_split['train'])]
    y_train = Y[0: int(n * train_val_test_split['train'])]
    
    X_test = X[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    y_test = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    
    X_val = X[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]
    y_val = Y[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]
    
    # Initialising the LSTM
    model = Sequential()
    
    # Adding layers. LSTM(n) --> Dropout(p)
    model.add(LSTM(units=10, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=int(d/ground_features), use_bias=False))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=int(d/ground_features), activation='linear', use_bias=True))
    
    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)
    
    # Compile
    model.compile(optimizer=adam_opt, loss=customized_loss)
    print(model.summary())
    
    # Fit
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    
    # Validate
    predicted_stock_returns = model.predict(X_val)
    predicted_stock_returns_test = model.predict(X_test)
    
    model_signal = []
    model_return = []
    market_return = []
    
    # Save
    pd.DataFrame(predicted_stock_returns).to_csv('C:/Users/Amit/src/output/LSTM_results/valid_results/all_stocks_pred.csv', index=False)
    pd.DataFrame(y_val).to_csv('C:/Users/Amit/src/output/LSTM_results/valid_results/all_stocks_real.csv', index=False)
    pd.DataFrame(predicted_stock_returns_test).to_csv('C:/Users/Amit/src/output/LSTM_results/test_results/all_stocks_pred.csv', index=False)
    pd.DataFrame(y_test).to_csv('C:/Users/Amit/src/output/LSTM_results/test_results/all_stocks_real.csv', index=False)
    
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig('loss_' + stocks[0])
    plt.close()
  
    for update_i in range(predicted_stock_returns[:,0].shape[0]):        
            if (update_i == 0):
                F_prev = np.sign(y_train[int(n * train_val_test_split['train'])-1][0])
                ## F_prev = np.sign(y_train[int(n * train_val_test_split['train'])-1][0])        
            F_prev_ = F_prev
            F_prev = np.sign(predicted_stock_returns[update_i][0])        
            model_signal.append(F_prev)
            temp_model_return = (F_prev_)*(y_val[update_i][0]) - 0.0002 * np.abs(F_prev_ - F_prev)        
            market_return.append(y_val[update_i][0])
            model_return.append(temp_model_return)
    
    return model_return, model_signal, market_return

tickers = ['MSFT','AMZN','FB','JPM','JNJ','PG','XOM','BA','NEE','AMT','APD']
weights = [0.273374590955138,0.172692307620917,0.114530395863918,0.101599971835774,0.0879559138286042,0.0744768071122344,0.0691212111706301,0.0441297912982564,0.0274199656770146,0.0226091544749039,0.0120898901626107]

model_return =[[] for i in range(len(tickers))]
model_signal = [[] for i in range(len(tickers))]
market_return = [[] for i in range(len(tickers))]

for i in range(len(tickers)):
    model_return_temp, model_signal_temp, market_return_temp = run_LSTM([tickers[i]])
    model_return[i] = model_return_temp
    model_signal[i] = model_signal_temp
    market_return[i] = market_return_temp

cum_model_return = [[weights[j]] for j in range(len(weights))]

for j in range(len(weights)):
    for i in range(len(model_return[j])):
        cum_model_return[j].append(cum_model_return[j][-1]*(1+(model_return[j][i])))

#truncating the results
limitN = 1000

cum_model_return = [cum_model_return[j][:limitN] for j in range(len(cum_model_return))]
cum_model_return2 = np.array(cum_model_return)
cum_model_return2 = np.sum(cum_model_return2, axis=0)

models_return = [model_return[j][:limitN] for j in range(len(model_return))]
models_return2 = np.array(models_return)
models_return2 = np.sum(models_return2, axis=0)


for j in range(len(weights)):
    print(np.shape(np.array(cum_model_return[j])))
    
cum_market_return = [[weights[j]] for j in range(len(weights))]

for j in range(len(weights)):
    for i in range(len(market_return[j])):
        cum_market_return[j].append(cum_market_return[j][-1]*(1+(market_return[j][i])))

#truncating the results
limitN = 1000

cum_market_return = [cum_market_return[j][:limitN] for j in range(len(cum_market_return))]
cum_market_return2 = np.array(cum_market_return)
cum_market_return2 = np.sum(cum_market_return2, axis=0)

markets_return = [market_return[j][:limitN] for j in range(len(market_return))]
markets_return2 = np.array(markets_return)
markets_return2 = np.sum(markets_return2, axis=0)


for j in range(len(weights)):
    print(np.shape(np.array(cum_market_return[j])))

plt.plot(cum_model_return2, label = 'Active long/short portfolio')
plt.plot(cum_market_return2, label = 'Long only portfolio')
plt.legend()


strategy_mean = np.mean(models_return2)
market_mean = np.mean(markets_return2)
print('Strategy Mean: ' , strategy_mean*100, '%')
print('Market Mean: ' , market_mean*100, '%')

strategy_std = np.std(models_return2)
market_std = np.std(markets_return2)
print('Strategy Std. Dev.: ' , strategy_std)
print('Market Std. Dev.: ' , market_std)

strategy_sharpe_ratio = strategy_mean / strategy_std
market_sharpe_ratio = market_mean / market_std
print('Strategy Sharpe Ratio: ' , strategy_sharpe_ratio)
print('Market Sharpe Ratio: ' , market_sharpe_ratio)