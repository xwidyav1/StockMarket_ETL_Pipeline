import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 14),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_sentiment_testing_4',
    default_args=default_args,
    description='A simple DAG to analyze stock sentiment',
    schedule_interval=timedelta(days=1),
    tags=['testing', 'stock'],
)

symbol = 'NVDA'
range_days = 15

def extract():
    load_dotenv()
    polygon_api_key = os.getenv('POLYGONIO_API_KEY')
    
    if not polygon_api_key:
        raise ValueError("API key is not set properly in the .env file.")
    
    stock_data = []
    news_data = []
    current_date = datetime.now()
    
    for i in range(range_days):
        date_str = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
        url_stock = f'https://api.polygon.io/v1/open-close/{symbol}/{date_str}?adjusted=true&apiKey={polygon_api_key}'
        response_stock = requests.get(url_stock)
        data_stock = response_stock.json()
        if 'status' in data_stock and data_stock['status'] == 'OK':
            stock_data.append(data_stock)
        
        url_news = f'https://api.polygon.io/v2/reference/news?ticker={symbol}&published_utc={date_str}&limit=10&apiKey={polygon_api_key}'
        response_news = requests.get(url_news)
        data_news = response_news.json()
        if 'results' in data_news:
            news_data.extend(data_news['results'])
    
    return stock_data, news_data

def process_stock_data(stock_data):
    df_stock_list = []
    for data in stock_data:
        df_stock = pd.DataFrame([data])
        df_stock['date'] = pd.to_datetime(df_stock['from'])
        df_stock.set_index('date', inplace=True)
        df_stock_list.append(df_stock[['open', 'high', 'low', 'close', 'volume']])
    
    df_stock_combined = pd.concat(df_stock_list)
    return df_stock_combined

def process_news_data(news_data, symbol):
    news_filtered = []
    for article in news_data:
        for insight in article.get('insights', []):
            if insight['ticker'] == symbol:
                news_filtered.append({
                    'published_utc': article['published_utc'],
                    'title': article['title'],
                    'sentiment': insight['sentiment'],
                    'sentiment_reasoning': insight['sentiment_reasoning']
                })
    return pd.DataFrame(news_filtered)

def transform(stock_data, news_data):
    df_stock = process_stock_data(stock_data)
    df_news = process_news_data(news_data, symbol)
    
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    if 'published_utc' in df_news.columns and 'title' in df_news.columns:
        news_df = df_news[['published_utc', 'title', 'sentiment', 'sentiment_reasoning']].copy()
        news_df['published_utc'] = pd.to_datetime(news_df['published_utc'])
        news_df['date'] = news_df['published_utc'].dt.date
        news_df['sentiment_scores'] = news_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
        
        sentiment_agg_nltk = news_df.groupby('date')['sentiment_scores'].mean().reset_index()
        sentiment_agg_nltk.columns = ['date', 'average_sentiment_nltk']
        
        sentiment_agg_insights = news_df.groupby('date')['sentiment'].apply(lambda x: x.mode()[0]).reset_index()
        sentiment_agg_insights.columns = ['date', 'average_sentiment_insights']
        
        sentiment_agg = pd.merge(sentiment_agg_nltk, sentiment_agg_insights, on='date')
        
        stock_df = df_stock.reset_index()
        stock_df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        stock_df['date'] = stock_df['datetime'].dt.date
        stock_df_details = stock_df[['date', 'high', 'low', 'close']].copy()
        
        final_df = pd.merge(stock_df[['date', 'close']], sentiment_agg, on='date', how='inner')
        final_df = final_df.sort_values('date')
        
        return final_df, sentiment_agg, sentiment_agg_nltk, news_df[['date', 'title', 'sentiment', 'sentiment_reasoning']], stock_df_details
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_to_postgresql(df, table_name):
    hook = PostgresHook(postgres_conn_id='stock_market_connection')
    engine = hook.get_sqlalchemy_engine()
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"DataFrame berhasil dimuat ke tabel '{table_name}' di database PostgreSQL.")

def extract_task(**kwargs):
    stock_data, news_data = extract()
    kwargs['ti'].xcom_push(key='stock_data', value=stock_data)
    kwargs['ti'].xcom_push(key='news_data', value=news_data)

def transform_task(**kwargs):
    stock_data = kwargs['ti'].xcom_pull(key='stock_data', task_ids='extract_task')
    news_data = kwargs['ti'].xcom_pull(key='news_data', task_ids='extract_task')
    final_df, sentiment_agg_df, sentiment_before_merge_df, news_details_df, stock_details_df = transform(stock_data, news_data)
    
    kwargs['ti'].xcom_push(key='final_df', value=final_df)
    kwargs['ti'].xcom_push(key='news_details_df', value=news_details_df)
    kwargs['ti'].xcom_push(key='stock_details_df', value=stock_details_df)
    
def load_task(**kwargs):
    final_df = kwargs['ti'].xcom_pull(key='final_df', task_ids='transform_task')
    news_details_df = kwargs['ti'].xcom_pull(key='news_details_df', task_ids='transform_task')
    stock_details_df = kwargs['ti'].xcom_pull(key='stock_details_df', task_ids='transform_task')
    
    load_to_postgresql(final_df, 'stock_sentiment')
    load_to_postgresql(news_details_df, 'news_details')
    load_to_postgresql(stock_details_df, 'stock_details')

extract_operator = PythonOperator(
    task_id='extract_task',
    python_callable=extract_task,
    provide_context=True,
    dag=dag,
)

transform_operator = PythonOperator(
    task_id='transform_task',
    python_callable=transform_task,
    provide_context=True,
    dag=dag,
)

load_operator = PythonOperator(
    task_id='load_task',
    python_callable=load_task,
    provide_context=True,
    dag=dag,
)

extract_operator >> transform_operator >> load_operator