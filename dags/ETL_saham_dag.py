from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import os
import requests
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'stock_sentiment_analysis',
    default_args=default_args,
    description='A simple DAG for stock sentiment analysis',
    schedule_interval=timedelta(days=1),
)

def extract_data():
    # Mengambil data dari News API
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY') 
    news_api_key = os.getenv('NEWS_API_KEY')    
    symbol = 'NVDA'  # Simbol saham NVIDIA
    q = 'NVIDIA'  # Query untuk pencarian berita
    upper_range = datetime.now().strftime('%Y-%m-%d')
    lower_range = (datetime.now() - timedelta(days=9)).strftime('%Y-%m-%d')
    url_stock = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_api_key}'
    url_news = f'https://newsapi.org/v2/everything?q={q}&from={lower_range}&to={upper_range}&sortBy=popularity&apiKey={news_api_key}'
    
    # Mengambil data dari API
    response_stock = requests.get(url_stock)
    data_stock = response_stock.json()

    response_news = requests.get(url_news)
    data_news = response_news.json()
    
    return data_stock, data_news

def process_data(ti):
    data_stock, data_news = ti.xcom_pull(task_ids='extract_data')
    # Proses data stock
    df_stock = process_stock_data(data_stock)

    # Mengubah JSON menjadi DataFrame untuk News API
    df_news = pd.json_normalize(data_news['articles']) if 'articles' in data_news else pd.DataFrame()

    # Transform data
    final_df = transform(df_stock, df_news)
    return final_df

def process_stock_data(data_stock):
    try:
        df_stock = pd.DataFrame.from_dict(data_stock['Time Series (Daily)'], orient='index')
        df_stock.index = pd.to_datetime(df_stock.index)
        df_stock.columns = ['open', 'high', 'low', 'close', 'volume']
        return df_stock
    except KeyError:
        print("Kunci 'Time Series (Daily)' tidak ditemukan dalam data JSON.")
        return pd.DataFrame()

def transform(df_stock, df_news):
    # Download vader lexicon for sentiment analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    # Extract necessary columns from news and create a copy
    news_df = df_news[['publishedAt', 'title']].copy()
    
    # Convert publishedAt to datetime
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    news_df['date'] = news_df['publishedAt'].dt.date
    
    # Sentiment analysis for each headline
    news_df['sentiment_scores'] = news_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Aggregate sentiment scores by date
    sentiment_agg = news_df.groupby('date')['sentiment_scores'].mean().reset_index()
    sentiment_agg.columns = ['date', 'average_sentiment']
    
    # Process stock data
    stock_df = df_stock.reset_index()
    stock_df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    stock_df['date'] = stock_df['datetime'].dt.date
    
    # Extract necessary columns from stock
    stock_df = stock_df[['date', 'close']].copy()
    
    # Merge sentiment and stock data by date
    final_df = pd.merge(stock_df, sentiment_agg, on='date', how='inner')
    
    # Sort by date
    final_df = final_df.sort_values('date')
    
    # Arrange column order
    final_df = final_df[['date', 'close', 'average_sentiment']]
    
    return final_df

def load_data(ti):
    final_df = ti.xcom_pull(task_ids='process_data')
    db_uri = os.getenv('DB_URI')
    engine = create_engine(db_uri)
    final_df.to_sql('stock_sentiment', engine, if_exists='replace', index=False)
    print("DataFrame berhasil dimuat ke tabel 'stock_sentiment' di database PostgreSQL.")

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Set task dependencies
extract_task >> process_task >> load_task
