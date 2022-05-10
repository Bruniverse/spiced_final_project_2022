import matplotlib.pyplot as plt
import yfinance as yf
from fbprophet import Prophet
import streamlit as st

st.title('Price Predictor')

ticker =  st.text_input('Type the name of the ticker or crypto','')

days =  st.text_input('Type how many days to create your prediction ','')
st.write(f'this is the price prediction for {ticker} for the next {days} days')

def get_df(ticker):
    df = yf.download(ticker, start='2017-01-01')
    df = df.reset_index()
    df[['ds','y']]= df[['Date', 'Adj Close']]
    return df

df = get_df(ticker)

def get_forecast(days, df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(int(days))
    forecast = model.predict(future)
    return forecast

def plot_pred(df, days):
    model = Prophet()
    model.fit(df)
    forecast = get_forecast(days, df)
    st.line_chart(forecast['yhat'])
    st.area_chart(forecast['yhat'])

plot_pred(df, days)

amount = st.text_input('Amount to investment:')

def investment_return(amount, days, df):
    current_price = df['Close'].iloc[[-1]].values
    model = Prophet()
    model.fit(df)
    forecast = get_forecast(days, df)
    predicted_price = forecast['yhat'].iloc[[-1]].values
    amount_return = (predicted_price * float(amount) / current_price)
    percentage = ((predicted_price - current_price) / current_price) * 100
    print(predicted_price,'hello.0', current_price,'hello.0.5', amount,'hello.1', amount_return)
    return amount_return, percentage, current_price, predicted_price

amount_return, percentage, current_price, predicted_price = investment_return(amount, days, df)
#st.write(amount_return[0], percentage[0], current_price[0], predicted_price)
st.write(f'- The current price is: $ {round(current_price[0], 2)}')
st.write(f'- The predicted price price is: $ {round(predicted_price[0], 2)}')
st.write(f'- The calculation of ROI is: {round(percentage[0], 2)} %')
st.write(f'- The calculation of total return investment is: $ {round(amount_return[0], 2)}')

st.write('By Bruno Saldivar, SPICED @ Data Science, Berlin May 2022.')

# ######## TO DO ########
#
#rerturn of investment
# ROI is calculated by subtracting the initial value of the investment
# from the final value of the investment (which equals the net return),
# then dividing this new number (the net return) by the cost of the investment,
# then finally, multiplying it by 100.

# documentation
#
# Plots:
# (https://docs.streamlit.io/library/api-reference/charts)
#
# ########       ########
