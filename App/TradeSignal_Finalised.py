# StreamLit setup
import streamlit as st
from PIL import Image

# Import libraries and dependencies
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# Timeseries packages
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

# MACD packages
import yfinance as yf
import vectorbt as vbt

# GIF
#st.image("Trading.gif", width=None)
st.image("Test.jpg", width=None)

st.title("Determine trade signals based upon MACD traces")

st.write('''
		 This app takes your stock ticker and provides you with a summary of the MACD signal
		 by.
		 ''')



st.header("Enter your stock ticker symbol")

ticker_symbol = st.text_input('(Example: AAPL)')

st.header("Enter amount you want to invest in this stock")
cash_inv = st.text_input('(Example: 1000)')


stock_name = ticker_symbol

if ticker_symbol:
    
    tickers = yf.download(tickers = stock_name,
                          start = "2020-01-01", 
                          period="max",                  
                          interval = "1d")
    
    st.header("Here is the tail of the main dataframe")
    st.table(tickers.tail())
    
    # Drop duplicates
    tickers.drop_duplicates(inplace=False)
    
    # Drop NA and infinite values
    tickers = tickers.replace(-np.inf, np.nan).dropna()
    
    
    # Plot ACF
    st.header('ACF plot is a plot that represents the correlation between a time series and lags of itself')
    plt.rcParams["figure.figsize"] = [10,5]
    st.pyplot(plot_acf(tickers.Close, lags=20))
    
    # Plot PACF
    st.header('PACF plot is a plot that represents the partial correlation between a time series and lags of itself')
    plt.rcParams["figure.figsize"] = [10,5]
    st.pyplot(plot_pacf(tickers.Close, zero=False, lags=20))
    
    # Get all combinations of [1, 2, 1]
    comb1 = combinations_with_replacement([1, 1, 1], 3)

    variables_list1 = []

    # Print the obtained combinations
    for i in list(comb1):
        if i[1] == 1: 
            variables_list1.append(i)

    
    
    # Get all combinations of [1, 1, 2] 
    comb2 = combinations_with_replacement([1, 1, 2], 3)

    variables_list2 = []

    # Print the obtained combinations
    for i in list(comb2):
        if i[1] == 1:            
            variables_list2.append(i)
            
    
    
    # Get all combinations of [1, 1, 3] 
#     comb3 = combinations_with_replacement([1, 1, 3], 3)
    
#     variables_list3 = []
    
#     # Print the obtained combinations
#     for i in list(comb3):
#         if i[1] == 1:    
#             variables_list3.append(i)


    
    # Get all combinations of [2, 1, 3]
#     comb4 = combinations_with_replacement([2, 1, 3], 3)

#     variables_list4 = []

#     # Print the obtained combinations
#     for i in list(comb4):
#         if i[1] == 1:            
#             variables_list4.append(i)


    variables_list = variables_list1 #+ variables_list2 #+ variables_list3 + variables_list4
    
    
    results_aic = []
    results_bic = []

    for combination in variables_list:
        model = ARIMA(tickers['Close'], 
                      order=combination)

        results = model.fit()

        results_aic.append(results.aic)
        results_bic.append(results.bic)

    
    
    df = pd.DataFrame({'aic':results_aic,
                       'bic':results_bic},
                      index=variables_list).sort_values(by='aic')
    
    aic_df_sorted = df.sort_values(by=['aic'], ascending=False)
    
    # Rerun ARIMA using optimised AIC (https://stats.stackexchange.com/questions/577/is-there-any-reason-to-prefer-the-aic-or-bic-over-the-other)
    model_refined = ARIMA(tickers['Close'],
                          order=aic_df_sorted.index[0])

    results_refined = model_refined.fit()

    st.text(results_refined.summary())
    
    
    
    # Forecast
    fc, se, conf = results_refined.forecast(5, alpha=0.05)  # 95% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=pd.date_range(tickers['Close'].index[-1] + datetime.timedelta(days=1), periods=5).tolist())
    lower_series = pd.Series(conf[:, 0], index=pd.date_range(tickers['Close'].index[-1] + datetime.timedelta(days=1), periods=5).tolist())
    upper_series = pd.Series(conf[:, 1], index=pd.date_range(tickers['Close'].index[-1] + datetime.timedelta(days=1), periods=5).tolist())
    

    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(tickers['Close'], label='Actual')
    plt.plot(fc_series, label='Forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    st.pyplot(plt)
    
    
    
    # Forecast close prices
    predicted_df = pd.DataFrame(results_refined.forecast(steps=5)[0],
                                index=pd.date_range(tickers['Close'].index[-1] + datetime.timedelta(days=1), periods=5).tolist())
    
    #plt.rcParams["figure.figsize"] = [10, 5]

    pred_plot = predicted_df.plot(title="5 Days Forecast")
    
    actual_close = tickers['Close']
    
    # Concat data
    prev_predit_df = pd.concat([actual_close, predicted_df],
                               axis=0)
    
    # Rename col header
    prev_predit_df.rename(columns={0: "Close"}, inplace=True)

    # Rename index
    prev_predit_df.index.names=["Date"]
    
    tickers_df = prev_predit_df.copy()    
    
    # Calculate MACD data and add to dictionary
    macd_list = {}
    

    # MACD data
    ewm_fast = tickers_df["Close"].ewm(span = 12, adjust = False).mean()
    ewm_slow = tickers_df["Close"].ewm(span = 26, adjust = False).mean()
    macd = pd.DataFrame(ewm_fast - ewm_slow)
    macd = macd.rename(columns = {"Close":"macd"})

    # Signal data
    signal = pd.DataFrame(macd["macd"].ewm(span = 9, adjust = False).mean()).rename(columns = {"macd":"signal"})

    # Histogram data
    histogram = pd.DataFrame(macd["macd"] - signal["signal"]).rename(columns = {0:("hist")})
    ticker_macd = pd.concat([macd, signal, histogram],
                            axis = 1)

    macd_list[ticker_symbol] = ticker_macd

        
        
    # Plot MACD data per individual stock
    
    plt.rcParams["figure.figsize"] = [18,12]

    ax1 = plt.subplot2grid((15,1), (0,0), rowspan = 5, colspan = 5)
    ax2 = plt.subplot2grid((15,1), (7,0), rowspan = 3, colspan = 5)
    ax1.plot(tickers_df["Close"], color = 'gray', linewidth = 2, label = ticker_symbol)
    ax1.set_title(f'{ticker_symbol} MACD SIGNALS')

    ax2.plot(macd_list[ticker_symbol]['macd'],
             color = 'skyblue',
             linewidth = 1.5, 
             label = 'MACD')

    ax2.plot(macd_list[ticker_symbol]['signal'],
             color = 'orange',
             linewidth = 1.5,
             label = 'SIGNAL')

    for i in range(len(macd_list[ticker_symbol])):

        if str(macd_list[ticker_symbol]['hist'][i])[0] == '-':

            ax2.bar(macd_list[ticker_symbol].index[i], 
                    macd_list[ticker_symbol]['hist'][i],
                    color = 'red')
        else:

            ax2.bar(macd_list[ticker_symbol].index[i], 
                    macd_list[ticker_symbol]['hist'][i], 
                    color = 'green')

    plt.legend(loc = 'lower right')
    st.pyplot(plt)
        
    def implement_macd_strategy(prices, data):  
        buy_price = []
        sell_price = []
        macd_signal = []
        signal = 0

        # For loop for range of dates
        for i in range(len(data)):

            # Conditional produce signal to buy stock
            if data['macd'][i] > data['signal'][i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)

            # Conditional produce signal to sell stock
            elif data['macd'][i] < data['signal'][i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        return buy_price, sell_price, macd_signal
    
    
    
     # Run trade strategy and plot buy and sell signals

    buy_price, sell_price, macd_signal = implement_macd_strategy(tickers_df["Close"], macd_list[ticker_symbol])

    plt.rcParams["figure.figsize"] = [18,12]

    ax1 = plt.subplot2grid((15,1), (0,0), rowspan = 5, colspan = 1)

    ax2 = plt.subplot2grid((15,1), (7,0), rowspan = 3, colspan = 1)

    ax1.plot(tickers_df["Close"], 
             color = 'gray',
             linewidth = 2,
             label = ticker_symbol)

    ax1.plot(tickers_df["Close"].index,
             buy_price, 
             marker = '^',
             color = 'green', 
             markersize = 6, 
             label = 'BUY SIGNAL',
             linewidth = 0)

    ax1.plot(tickers_df["Close"].index,
             sell_price, 
             marker = 'v', 
             color = 'r',
             markersize = 6,
             label = 'SELL SIGNAL',
             linewidth = 0)

    ax1.legend()

    ax1.set_title(f'{ticker_symbol} MACD BUY-SELL SIGNAL')

    ax2.plot(macd_list[ticker_symbol]['macd'],
             color = 'skyblue',
             linewidth = 1.5, 
             label = 'MACD')

    ax2.plot(macd_list[ticker_symbol]['signal'], 
             color = 'orange', 
             linewidth = 1.5,
             label = 'SIGNAL')

    for i in range(len(macd_list[ticker_symbol])):

        if str(macd_list[ticker_symbol]['hist'][i])[0] == '-':

            ax2.bar(macd_list[ticker_symbol].index[i],
                    macd_list[ticker_symbol]['hist'][i], 
                    color = 'r')
        else:

            ax2.bar(macd_list[ticker_symbol].index[i],
                    macd_list[ticker_symbol]['hist'][i], 
                    color = 'g')

    plt.legend(loc = 'lower right')
    st.pyplot(plt)
        
        
    # Run strategy to create position
    stock_strategy = {}

    buy_price, sell_price, macd_signal = implement_macd_strategy(tickers_df["Close"], macd_list[stock_name])

    position = []

    for i in range(len(macd_signal)):        
        if macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)

    for i in range(len(tickers_df['Close'])):

        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]

        macd = macd_list[stock_name]['macd']

        signal = macd_list[stock_name]['signal']

        close_price = tickers_df["Close"]

        macd_signal_df = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(tickers_df["Close"].index)

        position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(tickers_df["Close"].index)

        frames = [close_price, macd, signal, macd_signal_df, position]

        stock_strategy = pd.concat(frames, 
                                   join = 'inner',
                                   axis = 1)
        
    # Create dictionary and populate according to macd_signal
    stock_signals = {}

    for ticker in tickers:
        entries = []
        exits = []

        for sig in stock_strategy["macd_signal"]:

            if sig == -1:
                entries.append("False")
                exits.append("True")
            elif sig == 1:
                entries.append("True")
                exits.append("False")
            else:
                entries.append("False")
                exits.append("False")

        entries = pd.Series(entries,
                            index = tickers_df["Close"].index)

        exits = pd.Series(exits, index = tickers_df["Close"].index)

        # Change type to bool
        entries = entries == "True"
        exits = exits == "True"

        # Create dataframe
        entries = pd.DataFrame(entries).rename(columns = {0:'entries'}).set_index(tickers_df["Close"].index)

        exits = pd.DataFrame(exits).rename(columns = {0:'exits'}).set_index(tickers_df["Close"].index)

        close_price = tickers_df["Close"]

        frames = [close_price,
                  entries, 
                  exits]

        signals = pd.concat(frames, join = 'inner', axis = 1)

        stock_signals[ticker] = signals

       
    
    # Initial investment
    init_cash = int(cash_inv)

    # Total profit earned from holding stock. Timeframe 3 months
    for ticker in stock_name:

        price = vbt.YFData.download(ticker, start='2020-01-01').get('Close')

        portfolio = vbt.Portfolio.from_holding(price,
                                               init_cash = init_cash)

    portfolio.total_profit() 

    st.header(f"With an inital investment of ${init_cash} in this stock, your total profit of just holding the stock from 2020-01-01 will be ${portfolio.total_profit():.2f}")
    
    total_profit = []

    
    
    for ticker in tickers:
        # Build portfolio using macd signals
        portfolio = vbt.Portfolio.from_signals(stock_signals[ticker]["Close"],
                                               stock_signals[ticker]["entries"],
                                               stock_signals[ticker]["exits"],
                                               init_cash = init_cash)

        # Total profit
        total_profit.append(portfolio.total_profit())

    st.header(f"With an inital investment of ${init_cash} in this stock, the total profit of trading based upon MACD signals will be ${sum(total_profit):.2f}")