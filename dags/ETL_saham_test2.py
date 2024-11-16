from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import os
import requests
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'stock_sentiment_testing_2',
    default_args=default_args,
    description='A simple DAG for stock sentiment analysis using Polygon.io and News API',
    schedule_interval=timedelta(days=1),
    tags=['testing'],
)

def extract_data():
    load_dotenv()
    # Mengambil data dari News API
    polygon_api_key = os.getenv('POLYGONIO_API_KEY')
    news_api_key = os.getenv('NEWS_API_KEY')
    
    if not polygon_api_key or not news_api_key:
        raise ValueError("API keys are not set properly in the .env file.")
    
    symbol = 'NVDA'  # Simbol saham NVIDIA
    q = 'NVIDIA'  # Query untuk pencarian berita
    upper_range = datetime.now().strftime('%Y-%m-%d')
    lower_range = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')  # Perluas rentang tanggal
    url_stock = f'https://api.polygon.io/v1/open-close/{symbol}/2024-11-01?adjusted=true&apiKey={polygon_api_key}'
    url_news = f'https://newsapi.org/v2/everything?q={q}&from={lower_range}&to={upper_range}&sortBy=popularity&apiKey={news_api_key}'

    print(f'Polygon API: {url_stock}')
    print(f'News API: {url_news}')
    
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
    
    # Menyimpan hasil transformasi ke XComs
    ti.xcom_push(key='final_df', value=final_df.to_json())

def process_stock_data(data_stock):
    try:
        # Membuat DataFrame dari data JSON Polygon.io
        df_stock = pd.DataFrame([data_stock])
        df_stock['date'] = pd.to_datetime(df_stock['from'])
        df_stock.set_index('date', inplace=True)
        df_stock = df_stock[['open', 'high', 'low', 'close', 'volume']]
        return df_stock
    except KeyError:
        print("Kunci yang diperlukan tidak ditemukan dalam data JSON.")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong atau lakukan penanganan lain

def transform(df_stock, df_news):
    # Download vader lexicon for sentiment analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    # Extract necessary columns from news and create a copy
    if 'publishedAt' in df_news.columns and 'title' in df_news.columns:
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
    
    else:
        print("Kolom yang diperlukan tidak ditemukan dalam data berita.")
        return pd.DataFrame()

def load_data(ti):
    final_df_json_str = ti.xcom_pull(task_ids='process_data')
    
    if not final_df_json_str:
        raise ValueError("No data found in XComs.")
    
    # Mengubah JSON string menjadi DataFrame
    final_df_json_str = final_df_json_str[0] if isinstance(final_df_json_str, list) else final_df_json_str
    
    try:
        final_df = pd.read_json(final_df_json_str)
        
        if final_df.empty:
            raise ValueError("The DataFrame is empty.")
        
        # Membuat koneksi ke PostgreSQL menggunakan PostgresHook
        postgres_hook = PostgresHook(postgres_conn_id='stock_market_db')
        
        # Memuat DataFrame ke tabel PostgreSQL
        engine = postgres_hook.get_sqlalchemy_engine()
        
        final_df.to_sql('stock_sentiment', engine, if_exists='replace', index=False)
        
        print(f"DataFrame berhasil dimuat ke tabel 'stock_sentiment' di database PostgreSQL.")
    
    except ValueError as e:
        print(f"Error loading data: {e}")
    
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
   task_id='load_data',
   python_callable=load_data,
   provide_context=True,
   dag=dag,
)

# Set task dependencies
extract_task >> process_task >> load_task