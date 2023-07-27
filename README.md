# Alpha Analysts
#### Authored by: [Jenny](https://github.com/jennyntd), [Michelle](https://github.com/MishQ666), [Scott](https://github.com/Bomegolf), and [Thapa](https://github.com/TribThapa)


## Project Overview
Our project aimed to determine BUY-SELL signals for any stock listed on the US Stock Exchange using Time-series analysis. 

<p align="center">
    	<img src="https://techcrunch.com/wp-content/uploads/2019/06/GettyImages-1051659174.jpg?w=730&crop=1" width="1000">
</p>


*General Advice Warning*
*Information published on this website has been prepared for general information purposes only and not as specific advice to any particular person. Any advice contained in this document is General Advice and does not take into account any person’s particular investment objectives, financial situation and particular needs.*

*Before making an investment decision based on this advice you should consider, with or without the assistance of a qualified adviser, whether it is appropriate to your particular investment needs, objectives and financial circumstances.  Past performance of financial products is no assurance of future performance.*

*Due to time limitation, we are focusing on Amazon only for demonstration purpose.*


## Project description

- TimeSeries analysis was used to forecast the close price for any stock 5-days into the future
- We used the ARIMA model where:
    - ARIMA stands for auto-regressive integrated moving average.
    - It’s a way of modelling time series data for forecasting (i.e., for predicting future points in the series), in such a way that:
        - a pattern of growth/decline in the data is accounted for (hence the “auto-regressive” part)
        - the rate of change of the growth/decline in the data is accounted for (hence the “integrated” part)
        - noise between consecutive time points is accounted for (hence the “moving average” part)

- ARIMA models are typically expressed like “ARIMA(p,d,q)”, with the three terms p, d, and q defined as follows:
    - p means the number of preceding (“lagged”) Y values that have to be added/subtracted to Y in the model, so as to make better predictions based on local periods of growth/decline in our data. This captures the “autoregressive” nature of ARIMA.
    - d represents the number of times that the data have to be “differenced” to produce a stationary signal (i.e., a signal that has a constant mean over time). This captures the “integrated” nature of ARIMA. If d=0, this means that our data does not tend to go up/down in the long term (i.e., the model is already stationary”). In this case, then technically you are performing just ARMA, not AR-I-MA. If p is 1, then it means that the data is going up/down linearly. If p is 2, then it means that the data is going up/down exponentially. 

    - q represents the number of preceding/lagged values for the error term that are added/subtracted to Y. This captures the “moving average” part of ARIMA.

<p align="center">
    	<img src="/Image/ARIMA_Table.png" width="300" height="300">
</p>


<p align="center">
    	<img src="/Image/Forecast.JPG" width="500" height="300">
</p>


<p align="center">
    	<img src="/Image/ActualvPred.JPG" width="700" height="300">
</p>


- MACD indicators were used to determine bullish or bearish movement in the market to reflect stock price strengthening or weakening 

![TradeSignal](https://github.com/MishQ666/ProjectTwo-Alpha-Analysts/blob/main/Image/AMZN_MACDSig_Indv.JPG)

- Here, is the [MACD App](https://share.streamlit.io/tribthapa/tradepredictions/main/App/TradeSignal_Finalised.py) you can use to determine your BUY-SELL signal for any stock listed on the US Stock Exchange.


<p>&nbsp;</p>
 
Besides, we also used Classification and Regression machine learning models to train and predict stock prices:

Regression models:
- Regression
- Linear Regression
- Random Forest
- Extra trees
- Lasso Regression
- Ridge Regression
- Stochastic Gradient Design

In summary, all R squares have a negative value in the models selected above, indicating that the Regression models does not follow the trend of the data, so fits worse than a horizontal line. It is usually the case when there are constraints on either the intercept or the slope of the linear regression line.


<p align="center">
    	<img src="/Image/ML_Reg_Table.JPG" width="1000">
</p>
 
Classification models:
- Classification
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Ada Boost Classifier
- XGBoost Classifier

In summary, for all Classification models we have low recall and precision values. This suggests, on average, our Classification models would be ~65% accurate which is not good enough to determine BUY-SELL signals when trading.


<p align="center">
   	<img src="/Image/ML_Class_Table.JPG" width="1000">
</p>


<p>&nbsp;</p>

## Conclusion

In conclusion, MACD prices are good indicators to generate BUY-SELL signals. However, other metrics such as Relative Strength Indicator (RSI) and Fibonacci indicators should also be considered when making an informed decision to trigger a BUY or SELL.

<p>&nbsp;</p>


## Sources
- [Time-series Analysis 1](https://towardsdatascience.com/identifying-ar-and-ma-terms-using-acf-and-pacf-plots-in-time-series-forecasting-ccb9fd073db8)
- [Time-series Analysis 2](https://towardsdatascience.com/a-real-world-time-series-data-analysis-and-forecasting-121f4552a87)
- [ACF v PACF](https://people.duke.edu/~rnau/411arim3.htm)
- [Google Colab](https://drive.google.com/drive/folders/1abuvNk-AlsIswHqVwza9GbKKlGb1UYDL)
- [ScikitLearn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
- [Yfinance](https://pypi.org/project/yfinance/)
- [Vectorbt](https://vectorbt.dev/)
- [Quantstats](https://www.youtube.com/watch?v=gsS3JxPXXvg)











