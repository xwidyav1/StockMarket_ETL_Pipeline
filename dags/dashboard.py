import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import psycopg2
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Memuat environment variables dari file .env
load_dotenv()

# Membaca data dari PostgreSQL
db_uri = os.getenv('DB_URI')
if not db_uri:
    raise ValueError("DB_URI is not set properly in the environment variables.")
else:
    print(f"DB_URI is set to: {db_uri}")

try:
    engine = create_engine(db_uri)
    connection = engine.connect()
    print("Connection to PostgreSQL established successfully.")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    raise

df_stock_sentiment = pd.read_sql('SELECT * FROM stock_sentiment', connection)
df_news_details = pd.read_sql('SELECT * FROM news_details', connection)
df_stock_details = pd.read_sql('SELECT * FROM stock_details', connection)

# Inisialisasi aplikasi Dash
app = dash.Dash(__name__)

# Layout aplikasi
app.layout = html.Div(children=[
    html.H1(children='Stock Sentiment Dashboard'),

    dcc.Graph(
        id='stock-sentiment-graph',
        figure={
            'data': [
                go.Scatter(
                    x=df_stock_sentiment['date'],
                    y=df_stock_sentiment['close'],
                    mode='lines',
                    name='Close Price'
                ),
                go.Scatter(
                    x=df_stock_sentiment['date'],
                    y=df_stock_sentiment['average_sentiment_nltk'],
                    mode='lines',
                    name='Average Sentiment (NLTK)',
                    yaxis='y2'
                )
            ],
            'layout': go.Layout(
                title='Stock Price and Sentiment Over Time',
                yaxis=dict(title='Close Price'),
                yaxis2=dict(title='Average Sentiment', overlaying='y', side='right')
            )
        }
    ),

    dcc.Graph(
        id='news-sentiment-graph',
        figure={
            'data': [
                go.Bar(
                    x=df_news_details['date'],
                    y=df_news_details['sentiment'],
                    name='Sentiment'
                )
            ],
            'layout': go.Layout(
                title='News Sentiment Over Time',
                yaxis=dict(title='Sentiment')
            )
        }
    ),

    dcc.Graph(
        id='stock-details-graph',
        figure={
            'data': [
                go.Candlestick(
                    x=df_stock_details['date'],
                    open=df_stock_details['open'],
                    high=df_stock_details['high'],
                    low=df_stock_details['low'],
                    close=df_stock_details['close'],
                    name='Stock Details'
                )
            ],
            'layout': go.Layout(
                title='Stock Details Over Time',
                yaxis=dict(title='Price')
            )
        }
    )
])

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run_server(debug=True)