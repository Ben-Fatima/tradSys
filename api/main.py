import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import os
import glob
import time
import json
import sys

import torch
import torch.nn as nn
import torch.optim as opt

import math
from sklearn.metrics import mean_squared_error



##############################################################################
#                                   Get Data                                 #
##############################################################################

##### Download data
#####    Input: tick which is a string     e.g.  MSFT, AAPL, MSFT, ...
#####    Output: a tuple of two values
#####        bool:          no errors occured (True) otherwise (False)
#####        pd.DataFrame:  data frame or None in case of errors
def download_data(tick: str) -> Tuple[bool, pd.DataFrame]:
    try:
        files = os.listdir('public/data')
        for file in files:
            if os.path.isfile('public/data'+file) and file == tick+'.csv':
                no_err, df = upload_data('public/data/'+tick+'.csv')
                return no_err, df

        data = yf.Ticker(tick)
        df = data.history(period='1d', start='2010-01-01')
        df.to_csv(os.path.join('public/data', tick+'.csv'))
        return True, df
    except Exception:
        return False, None

##### Upload data
#####    Input: path to csv file which is a string     e.g.  public/MSFT.csv
#####    Output: a tuple of two values
#####        bool:          no errors occured (True) otherwise (False)
#####        pd.DataFrame:  data frame or None in case of errors
def upload_data(path: str) -> Tuple[bool, pd.DataFrame]:
    try:
        df = pd.read_csv(path).set_index('Date')
        return True, df
    except Exception:
        return False, None

##############################################################################
#                                 Training a Model                           #
##############################################################################
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def train(tick: str, data: pd.DataFrame, arch: str='LSTM', h_dim: int=32, n_layers: int=2, lr: float=.01, n_epochs: int=100, horizon: int=30, train_ratio: float=.8) -> bool:
    try:
        clear_console()
        log_console('Preparing Data...')

        price = data[['Close']]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

        def split_data(stock, lookback, train_ratio):
            data_raw = stock.to_numpy() # convert to numpy array
            data = []
            
            # create all possible sequences of length seq_len
            for index in range(len(data_raw) - lookback): 
                data.append(data_raw[index: index + lookback])
            
            data = np.array(data);
            test_set_size = int(np.round((1-train_ratio)*data.shape[0]));
            train_set_size = data.shape[0] - (test_set_size);
            
            x_train = data[:train_set_size,:-1,:]
            y_train = data[:train_set_size,-1,:]
            
            x_test = data[train_set_size:,:-1]
            y_test = data[train_set_size:,-1,:]
            
            return [x_train, y_train, x_test, y_test]

        x_train, y_train, x_test, y_test = split_data(price, horizon, train_ratio)

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)
        log_console('Data prerpartion complete')

        input_dim = 1
        output_dim = 1

        if arch == 'LSTM':
            model = LSTM(input_dim=input_dim, hidden_dim=h_dim, output_dim=output_dim, num_layers=n_layers)
        elif arch == 'GRU':
            model = GRU(input_dim=input_dim, hidden_dim=h_dim, output_dim=output_dim, num_layers=n_layers)
        else:
            raise Exception('Architecture not rcognized')

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        hist = np.zeros(n_epochs)
        start_time = time.time()
        lstm = []
        for t in range(n_epochs):
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            log_console("Epoch {:4.0f}, MSE: {:.8f}".format(t, loss.item()))
            hist[t] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_time = time.time()-start_time
        log_console("Training time: {}".format(training_time))

        log_console('Evaluating the trained model...')
        y_test_pred = model(x_test)
        
        # invert predictions
        y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
        y_train = scaler.inverse_transform(y_train.detach().numpy())
        y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
        y_test = scaler.inverse_transform(y_test.detach().numpy())
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
        log_console('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
        log_console('Test Score: %.2f RMSE' % (testScore))
        lstm.append(trainScore)
        lstm.append(testScore)
        lstm.append(training_time)

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(price)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[horizon:len(y_train_pred)+horizon, :] = y_train_pred

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(price)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(y_train_pred)+horizon-1:len(price)-1, :] = y_test_pred

        original = scaler.inverse_transform(price['Close'].values.reshape(-1,1))

        predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
        predictions = np.append(predictions, original, axis=1)
        result = pd.DataFrame(predictions)
        
        plt.figure(figsize=(15,5))
        plt.plot(result.index, result[2], label='Actual value', alpha=.5)
        plt.plot(result.index, result[0], label='Train prediction')
        plt.plot(result.index, result[1], label='Test prediction')
        plt.legend()
        plt.title(tick+' Model Evaluation')
        
        plt.savefig('public/figure.png')
        log_console('Evaluation complete...')
        ########################################################################
        # model.eval()

        # batch = data['Close'].values
        # #batch = batch[-horizon:].reshape((1, horizon, 1))
        # steps = 100
        # pred_list = []
        # for i in range(steps): 
        #     batch = scaler.fit_transform(batch.reshape(-1,1)) 
        #     batch = batch[-steps:].reshape((1, steps, 1)) 
            
        #     batch = torch.from_numpy(batch).type(torch.Tensor)
        #     pred_list.append(model(batch).item()) 
        #     batch = np.append(
        #         batch[:,1:,:],
        #         [[[pred_list[i]]]],
        #         axis=1
        #     )
        #     batch = scaler.inverse_transform(batch.reshape((steps, 1)))

        # pred_list = scaler.inverse_transform(np.array(pred_list).reshape((steps, 1)))
        # orig_lisr = scaler.inverse_transform(np.array(price['Close'].values).reshape((price.shape[0], 1)))

        # orig_data = np.empty((price.shape[0]+steps, 1))
        # orig_data[:,:] = np.nan
        # orig_data[:len(price), 0] = orig_lisr[:, 0]

        # pred_data = np.empty_like(orig_data)
        # pred_data[:,:] = np.nan
        # print(pred_list[:, 0].shape)
        # pred_data[len(price):, 0] = pred_list[:, 0]

        # dataa = np.append(orig_data, pred_data, axis=1)
        # result = pd.DataFrame(dataa)

        # plt.figure(figsize=(15,5))
        # plt.plot(result.index, result[1], label='Past')
        # plt.plot(result.index, result[0], label='Future')
        # plt.legend()
        # plt.title(tick)
        
        # plt.savefig('public/'+tick+'_pred.png')
        ################################################################################################

        torch.save(
            {
                'arch': arch,
                'inp_dim': input_dim,
                'h_dim': h_dim,
                'out_dim': output_dim,
                'n_layers': n_layers,
                'horizon': horizon,
                'train_ratio': train_ratio,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'criterion': criterion,
                'scaler': scaler
            },
            os.path.join('public/models', tick+'.pth')
        )

        return True
    except Exception as e:
        raise e
        return False

##############################################################################
#                                 Set Indicators                             #
##############################################################################
class DMAC:
    def __init__(self, tick: str, df: pd.DataFrame, initial_cap: float=10_000.0) -> None:
        self.tick = tick
        self.data = pd.DataFrame()
        self.data['Close'] = df['Close']
        self.data['SMA30'] = df['Close'].rolling(window=30).mean()
        self.data['SMA100'] = df['Close'].rolling(window=100).mean()
        self.capital = initial_cap

        self.buy_sell()
        self.plot()
        self.inspect()
        
    def buy_sell(self):
        sigPriceBuy = []
        sigPriceSell = []
        flag = -1

        for i in range(len(self.data)):
            if self.data['SMA30'][i] > self.data['SMA100'][i]:
                if flag != 1:
                    sigPriceBuy.append(self.data['Close'][i])
                    sigPriceSell.append(np.nan)
                    flag = 1
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            elif self.data['SMA30'][i] < self.data['SMA100'][i]:
                if flag != 0:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(self.data['Close'][i])
                    flag = 0
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)

        self.data['Buy'] = sigPriceBuy
        self.data['Sell'] = sigPriceSell
        
    def plot(self):
        plt.figure(figsize=(15,5))
        plt.plot(self.data['Close'], label='aapl', alpha=0.35)
        plt.plot(self.data['SMA30'], label='SMA30', alpha=0.7)
        plt.plot(self.data['SMA100'], label='SMA100', alpha=0.7)
        plt.scatter(self.data.index, self.data['Buy'], label='Buy', marker='^', color='green')
        plt.scatter(self.data.index, self.data['Sell'], label='Sell', marker='v', color='red')
        plt.title(self.tick + "Close Price History")
        plt.xlabel('date')
        plt.ylabel('Close Price')
        plt.legend(loc='upper left')
        plt.savefig('public/figure.png')


    def inspect(self):
        clear_console()
        for i in range(len(self.data)):
            if not np.isnan(self.data['Buy'].iloc[i]):
                self.capital -= self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : BUY  :  {self.capital:5.6f}")
            if not np.isnan(self.data['Sell'].iloc[i]):
                self.capital += self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : SELL :  {self.capital:5.6f}")

class OBV:
    def __init__(self, tick: str, df: pd.DataFrame, initial_cap: float=10_000.0) -> None:
        self.tick = tick
        self.data = pd.DataFrame()
        self.data['Close'] = df['Close']
        self.data['Volume'] = df['Volume']
        
        obv = [0]
        for i in range(1, len(self.data['Close'])):
            if self.data['Close'][i] > self.data["Close"][i-1]:
                obv.append(obv[-1] + self.data['Volume'][i])
            elif self.data['Close'][i] < self.data['Close'][i-1]:
                obv.append(obv[-1] - self.data['Volume'][i])
            else:
                obv.append(obv[-1])

        self.data['OBV'] = obv
        self.data['OBV_EMA'] = self.data['OBV'].ewm(span=20).mean()
        self.capital = initial_cap

        self.buy_sell()
        self.plot()
        self.inspect()
        
    def buy_sell(self):
        sigPriceBuy = []
        sigPriceSell = []
        flag = -1

        for i in range(0, len(self.data)):
            if self.data['OBV'][i] > self.data['OBV_EMA'][i] and flag != 1:
                sigPriceBuy.append(self.data['Close'][i])
                sigPriceSell.append(np.nan)
                flag = 1
            elif self.data['OBV'][i] < self.data['OBV_EMA'][i] and flag != 0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(self.data['Close'][i])
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)

        self.data['Buy'] = sigPriceBuy
        self.data['Sell'] = sigPriceSell
        
    def plot(self):
        fig, ax = plt.subplots(2, 1, figsize=(15, 5))
        ax[0].plot(self.data['Close'], label='aapl', alpha=0.35)
        ax[1].plot(self.data['OBV'], label='OBV', alpha=1)
        ax[1].plot(self.data['OBV_EMA'], label='OBV_EMA', alpha=1)
        ax[0].scatter(self.data.index, self.data['Buy'], label='Buy', marker='^', color='green')
        ax[0].scatter(self.data.index, self.data['Sell'], label='Sell', marker='v', color='red')
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper left')
        fig.savefig('public/figure.png')

    def inspect(self):
        clear_console()
        for i in range(len(self.data)):
            if not np.isnan(self.data['Buy'].iloc[i]):
                self.capital -= self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : BUY  :  {self.capital:5.6f}")
            if not np.isnan(self.data['Sell'].iloc[i]):
                self.capital += self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : SELL :  {self.capital:5.6f}")

class RSI:
    def __init__(self, tick: str, df: pd.DataFrame, initial_cap: float=10_000.0) -> None:
        self.tick = tick
        self.data = pd.DataFrame()
        self.data['Close'] = df['Close']

        delta = self.data['Close'].diff(1).dropna()
        loss = delta.copy()
        gains = delta.copy()

        gains[gains<0] = 0
        loss[loss>0] = 0

        gain_ewm = gains.ewm(com=13).mean()
        loss_ewm = abs(loss.ewm(com=13).mean())

        RS = gain_ewm / loss_ewm
        self.data['RSI'] = 100 - 100/(1 + RS)
        self.capital = initial_cap

        self.buy_sell()
        self.plot()
        self.inspect()
        
    def buy_sell(self):
        self.data['Buy'] = np.where((self.data['RSI'].values > 70) & (self.data['RSI'].shift(1) <= 70), self.data['Close'], np.nan)
        self.data['Sell'] = np.where((self.data['RSI'].values < 30) & (self.data['RSI'].shift(1) >= 30), self.data['Close'], np.nan)
        
    def plot(self):
        fig, ax = plt.subplots(2, 1, figsize=(15, 5))
        ax[0].plot(self.data['Close'], label='aapl', alpha=0.35)
        ax[0].scatter(self.data.index, self.data['Buy'], label='Buy', marker='^', color='green')
        ax[0].scatter(self.data.index, self.data['Sell'], label='Sell', marker='v', color='red')
        ax[0].legend(loc='upper left')
        ax[1].plot(self.data['RSI'], label='RSI', c='purple')
        ax[1].axhline(y=70, c='blue', label='Overbought')
        ax[1].axhline(y=30, c='orange', label='Oversold')
        ax[1].legend(loc='lower right', fontsize=8)
        fig.savefig('public/figure.png')

    def inspect(self):
        clear_console()
        for i in range(len(self.data)):
            if not np.isnan(self.data['Buy'].iloc[i]):
                self.capital -= self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : BUY  :  {self.capital:5.6f}")
            if not np.isnan(self.data['Sell'].iloc[i]):
                self.capital += self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : SELL :  {self.capital:5.6f}")

def smma(df, period, column_name, shift):
    """Calculate Smoothed Moving Average"""
    df_tmp = pd.DataFrame()
    df_tmp['Median'] = (df["High"] + df["Low"])/ 2
    first_val = df_tmp['Median'].iloc[:period].mean()
    df_tmp[column_name] = np.nan
    df_tmp[column_name].iloc[period] = first_val
    for i in range(len(df_tmp)):
        if i > period:
            smma_val = (df_tmp[column_name].iloc[i-1] * (period - 1) + df_tmp['Median'].iloc[i]) / period
            df_tmp[column_name].iloc[i] = smma_val
    df_tmp = df_tmp[[column_name]]
    return df_tmp.shift(shift)

class Alligator:
    def __init__(self, tick: str, df: pd.DataFrame, initial_cap: float=10_000.0) -> None:
        self.tick = tick
        self.data = pd.DataFrame()
        self.data['High'] = df['High']
        self.data['Low'] = df['Low']
        self.data['Close'] = df['Close']
        
        self.data['jaws'] = smma(self.data, 13, "jaws", 8)
        self.data['teeth'] = smma(self.data, 8, "teeth", 5)
        self.data['lips'] = smma(self.data, 5, "lips", 3)
        self.capital = initial_cap
        print("a")
        self.buy_sell()
        self.plot()
        self.inspect()
        print("b")
    def buy_sell(self):
        sigPriceBuy = [np.nan] * self.data.shape[0]
        sigPriceSell = [np.nan] * self.data.shape[0]

        prev_f = self.data['lips'].shift(1)
        prev_s = self.data['jaws'].shift(1)

        for i in range(len(self.data)):
            if self.data['lips'].iloc[i] <= self.data['jaws'].iloc[i] and prev_f[i] >= prev_s[i]:
                sigPriceSell[i] = self.data['Close'][i]
            if self.data['lips'].iloc[i] >= self.data['jaws'].iloc[i] and prev_f[i] <= prev_s[i]:
                sigPriceBuy[i] = self.data['Close'][i]
                
        self.data['Buy'] = sigPriceBuy
        self.data['Sell'] = sigPriceSell
        
    def plot(self):
        plt.figure(figsize=(15,5))
        plt.plot(self.data['jaws'], label='jaws', c='b', alpha=0.5)
        plt.plot(self.data['teeth'], label='teeth', c='tomato', alpha=0.5)
        plt.plot(self.data['lips'], label='lips', c='darkgreen', alpha=0.5)
        plt.scatter(self.data.index, self.data['Buy'], label='Buy', marker='^', color='green')
        plt.scatter(self.data.index, self.data['Sell'], label='Sell', marker='v', color='red')
        plt.title("Apple Adj. Close Price History")
        plt.xlabel('date')
        plt.ylabel('Adj. Close Price')
        plt.legend(loc='upper left')
        plt.savefig('public/figure.png')

    def inspect(self):
        clear_console()
        for i in range(len(self.data)):
            if not np.isnan(self.data['Buy'].iloc[i]):
                self.capital -= self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : BUY  :  {self.capital:5.6f}")
            if not np.isnan(self.data['Sell'].iloc[i]):
                self.capital += self.data['Close'].iloc[i]
                log_console(f"{self.data.index[i]} : SELL :  {self.capital:5.6f}")


##############################################################################
#                                    Helpers                                 #
##############################################################################
def log_console(line: str) -> None:
    if not os.path.isfile('public/console.txt'):
        open('public/console.txt', 'w')
    with open('public/console.txt', 'a') as console:
        console.write(line+'\n')

def clear_console() -> None:
    open('public/console.txt', 'w')

def main():
    #if inputs == None:
    inputs = json.loads(sys.stdin.read())

    tick = inputs['data']
    no_errors, df = download_data(tick)

    if inputs['method'] == "train":
        print(df.head())
        train(
            tick, 
            df, 
            inputs['arch'], 
            int(inputs['h_dim']),
            int(inputs['n_layers']),
            float(inputs['lr']),
            int(inputs['n_epoques']),
            int(inputs['horizon'])
        )
    elif inputs['method'] == "pred":
        if inputs['algo'] == "dmac":
            DMAC(tick, df)
        elif inputs['algo'] == 'obv':
            OBV(tick, df)
        elif inputs['algo'] == 'Rsi':
            RSI(tick, df)
        elif inputs['algo'] == 'alligator':
            Alligator(tick, df)

if __name__ == '__main__':
    main()

 