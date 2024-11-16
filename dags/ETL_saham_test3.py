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
    'stock_testing_3bangmantap',
    default_args=default_args,
    description='A simple DAG for stock sentiment analysis using Polygon.io and News API',
    schedule_interval=timedelta(days=1),
    tags=['testing'],
)

def extract_data(ti):
    load_dotenv()
    # Mengambil data dari News API
    polygon_api_key = os.getenv('POLYGONIO_API_KEY')
    news_api_key = os.getenv('NEWS_API_KEY')
    
    if not polygon_api_key or not news_api_key:
        raise ValueError("API keys are not set properly in the .env file.")
    
    symbol = 'NVDA'  # Simbol saham NVIDIA
    q = 'NVIDIA'  # Query untuk pencarian berita
    range_days = 15
    upper_range = datetime.now().strftime('%Y-%m-%d')
    lower_range = (datetime.now() - timedelta(days=range_days)).strftime('%Y-%m-%d')  # Perluas rentang tanggal
    
    # Mengambil data saham dari Polygon.io dalam rentang tanggal tertentu
    stock_data = []
    current_date = datetime.now()
    for i in range(range_days):
        date_str = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
        url_stock = f'https://api.polygon.io/v1/open-close/{symbol}/{date_str}?adjusted=true&apiKey={polygon_api_key}'
        response_stock = requests.get(url_stock)
        data_stock = response_stock.json()
        if 'status' in data_stock and data_stock['status'] == 'OK':
            stock_data.append(data_stock)
    
    url_news = f'https://newsapi.org/v2/everything?q={q}&from={lower_range}&to={upper_range}&sortBy=popularity&apiKey={news_api_key}'

    print(f'Polygon API: {url_stock}')
    print(f'News API: {url_news}')
    
    # Mengambil data dari News API
    response_news = requests.get(url_news)
    data_news = response_news.json()
    
    ti.xcom_push(key='data_stock', value=stock_data)
    ti.xcom_push(key='data_news', value=data_news)

def process_data(ti):
    stock_data = ti.xcom_pull(task_ids='extract_data', key='data_stock')
    data_news = ti.xcom_pull(task_ids='extract_data', key='data_news')
    
    if not stock_data or not data_news:
        raise ValueError("No data found in XComs.")
    
    # Proses data stock
    df_stock = process_stock_data(stock_data)

    # Mengubah JSON menjadi DataFrame untuk News API
    df_news = pd.json_normalize(data_news['articles']) if 'articles' in data_news else pd.DataFrame()

    # Transform data
    final_df, df_news_transformed, df_stock_transformed = transform(df_stock, df_news)
    
    # Menyimpan hasil transformasi ke XComs
    ti.xcom_push(key='final_df', value=final_df.to_json())
    ti.xcom_push(key='df_stock_transformed', value=df_stock_transformed.to_json())
    ti.xcom_push(key='df_news_transformed', value=df_news_transformed.to_json())

def process_stock_data(stock_data):
    try:
        # Membuat DataFrame dari data JSON Polygon.io untuk rentang tanggal tertentu
        df_stock_list = []
        for data in stock_data:
            df_stock = pd.DataFrame([data])
            df_stock['date'] = pd.to_datetime(df_stock['from'])
            df_stock.set_index('date', inplace=True)
            df_stock_list.append(df_stock[['open', 'high', 'low', 'close']])
        
        df_stock_combined = pd.concat(df_stock_list)
        return df_stock_combined
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
        news_df['sentiment'] = news_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
        
        # Extract necessary columns from stock and news without aggregation
        stock_df_transformed = df_stock.reset_index()
        stock_df_transformed.columns = ['date', 'open', 'high', 'low', 'close']
        
        news_df_transformed = news_df[['date', 'title', 'sentiment']]
        
        # Ensure both date columns are of the same type before merging
        stock_df_transformed['date'] = pd.to_datetime(stock_df_transformed['date'])
        news_df_transformed['date'] = pd.to_datetime(news_df_transformed['date'])
        
        # Merge sentiment and stock data by date for final_df
        final_df = pd.merge(stock_df_transformed[['date', 'close']], news_df_transformed, on='date', how='inner')
        
        return final_df, news_df_transformed, stock_df_transformed
    
    else:
        print("Kolom yang diperlukan tidak ditemukan dalam data berita.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_data(ti):
    final_df_json_str = ti.xcom_pull(task_ids='process_data', key='final_df')
    df_stock_json_str = ti.xcom_pull(task_ids='process_data', key='df_stock_transformed')
    df_news_json_str = ti.xcom_pull(task_ids='process_data', key='df_news_transformed')
    
    if not final_df_json_str or not df_stock_json_str or not df_news_json_str:
        raise ValueError("No data found in XComs.")
    
    try:
        final_df = pd.read_json(final_df_json_str)
        df_stock_transformed = pd.read_json(df_stock_json_str)
        df_news_transformed = pd.read_json(df_news_json_str)
        
        if final_df.empty or df_stock_transformed.empty or df_news_transformed.empty:
            raise ValueError("The DataFrame is empty.")
        
        # Membuat koneksi ke PostgreSQL menggunakan PostgresHook
        postgres_hook = PostgresHook(postgres_conn_id='stock_market_connection')
        
        # Memuat DataFrame ke tabel PostgreSQL
        engine = postgres_hook.get_sqlalchemy_engine()
        
        final_df.to_sql('stock_sentiment', engine, if_exists='replace', index=False)
        df_stock_transformed.to_sql('stock_data', engine, if_exists='replace', index=False)
        df_news_transformed.to_sql('news_data', engine, if_exists='replace', index=False)
        
        print(f"DataFrame berhasil dimuat ke tabel 'stock_sentiment', 'stock_data', dan 'news_data' di database PostgreSQL.")
    
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