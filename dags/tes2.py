import os
from dotenv import load_dotenv
import requests
from pprint import pprint
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from sqlalchemy import create_engine

def extract():
    load_dotenv()
    # Mengambil data dari News API
    polygon_api_key = os.getenv('POLYGONIO_API_KEY')
    news_api_key = os.getenv('NEWS_API_KEY')
    
    if not polygon_api_key or not news_api_key:
        raise ValueError("API keys are not set properly in the .env file.")
    
    symbol = 'NVDA'  # Simbol saham NVIDIA
    q = 'NVIDIA'  # Query untuk pencarian berita
    range_days = 5
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
    
    return stock_data, data_news

# Menjalankan fungsi extract dan mencetak respon dari API dengan pprint
stock_data, data_news = extract()
print("Response from Polygon.io API (Stock Data):")
pprint(stock_data)
print("\nResponse from News API (News Data):")
pprint(data_news)

# Mengubah JSON menjadi DataFrame untuk Polygon.io
def process_stock_data(stock_data):
    try:
        # Membuat DataFrame dari data JSON Polygon.io untuk rentang tanggal tertentu
        df_stock_list = []
        for data in stock_data:
            df_stock = pd.DataFrame([data])
            df_stock['date'] = pd.to_datetime(df_stock['from'])
            df_stock.set_index('date', inplace=True)
            df_stock_list.append(df_stock[['open', 'high', 'low', 'close', 'volume']])
        
        df_stock_combined = pd.concat(df_stock_list)
        return df_stock_combined
    except KeyError:
        print("Kunci yang diperlukan tidak ditemukan dalam data JSON.")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong atau lakukan penanganan lain

# Mengubah JSON menjadi DataFrame untuk News API
df_news = pd.json_normalize(data_news['articles']) if 'articles' in data_news else pd.DataFrame()

# Proses data stock
df_stock = process_stock_data(stock_data)

print("\nDataFrame dari Polygon.io API (Stock Data):")
print(df_stock)

print("\nDataFrame dari News API (News Data):")
print(df_news)

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
        
        # Debugging: Print intermediate DataFrames
        print("\nStock DataFrame before merge:")
        print(stock_df)
        print("\nSentiment DataFrame before merge:")
        print(sentiment_agg)
        
        # Merge sentiment and stock data by date
        final_df = pd.merge(stock_df, sentiment_agg, on='date', how='inner')
        
        # Debugging: Print final DataFrame after merge
        print("\nFinal DataFrame after merge:")
        print(final_df)
        
        # Sort by date
        final_df = final_df.sort_values('date')
        
        # Arrange column order
        final_df = final_df[['date', 'close', 'average_sentiment']]
        
        return final_df
    
    else:
        print("Kolom yang diperlukan tidak ditemukan dalam data berita.")
        return pd.DataFrame()

# Example usage
final_df = transform(df_stock, df_news)
print("\nAggregated Sentiment DataFrame:")
print(final_df)

# def load_to_postgresql(df, table_name, db_uri):
#     """
#     Memuat DataFrame ke database PostgreSQL.

#     Parameters:
#     df (DataFrame): DataFrame yang akan dimuat ke database.
#     table_name (str): Nama tabel di database.
#     db_url (str): URL koneksi ke database PostgreSQL.

#     Returns:
#     None
#     """
#     # Membuat engine koneksi ke PostgreSQL
#     engine = create_engine(db_uri)
    
#     # Memuat DataFrame ke tabel PostgreSQL
#     df.to_sql(table_name, engine, if_exists='replace', index=False)
    
#     print(f"DataFrame berhasil dimuat ke tabel '{table_name}' di database PostgreSQL.")

# # Contoh penggunaan fungsi load_to_postgresql
# db_uri = os.getenv('DB_URI')
# load_to_postgresql(final_df, 'stock_sentiment', db_uri)