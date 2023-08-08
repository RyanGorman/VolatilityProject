import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request

API_KEY = 'N0OEN8EEDJH5ZO60'


def get_stock_data(symbol):
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(api_url)
    data = response.json()

    if 'Time Series (Daily)' not in data:
        raise ValueError(f"No data available for symbol: {symbol}")

    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df = df.sort_index()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df


def calculate_volatility(df):
    df['Log Returns'] = np.log(df['Close'] / df['Close'].shift())
    df['Volatility'] = df['Log Returns'].rolling(window=5).std() * np.sqrt(252)

    # Calculate Garman-Klass-Yang-Zhang volatility estimator
    window = 5
    df['Garman-Klass-Yang-Zhang Volatility'] = np.sqrt(252 / window * pd.DataFrame.rolling(
        np.log(df['Open'] / df['Close'].shift(1)) ** 2 +
        0.5 * np.log(df['High'] / df['Low']) ** 2 -
        (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2, window=window).sum())

    df.drop(['Log Returns'], axis=1, inplace=True)
    return df


def create_plot(data, symbol):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Price'), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], name='Volatility (Rolling)'), secondary_y=True)
    fig.add_trace(go.Scatter(x=data.index, y=data['Garman-Klass-Yang-Zhang Volatility'],
                             name='Garman-Klass-Yang-Zhang Volatility'), secondary_y=True)

    fig.update_layout(title=f"{symbol} Stock Volatility", xaxis_title="Date",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    # Set y-axis titles for primary and secondary y-axes
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volatility", secondary_y=True)

    return fig.to_html(full_html=False)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symbol = request.form['symbol']
        data = get_stock_data(symbol)
        data = calculate_volatility(data)
        plot = create_plot(data, symbol)
        return render_template('index.html', plot=plot)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()

