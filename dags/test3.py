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
    # Mengambil data dari Polygon.io API
    polygon_api_key = os.getenv('POLYGONIO_API_KEY')
    
    if not polygon_api_key:
        raise ValueError("API key is not set properly in the .env file.")
    
    symbol = 'NVDA'  # Simbol saham NVIDIA
    range_days = 10
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
    
    # Mengambil data berita dari Polygon.io dalam rentang tanggal tertentu
    news_data = []
    for i in range(range_days):
        date_str = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
        url_news = f'https://api.polygon.io/v2/reference/news?ticker={symbol}&published_utc={date_str}&limit=20&apiKey={polygon_api_key}'
        response_news = requests.get(url_news)
        data_news = response_news.json()
        if 'results' in data_news:
            news_data.extend(data_news['results'])

    print(f'Polygon API: {url_stock}')
    print(f'News API: {url_news}')
    
    return stock_data, news_data

# Menjalankan fungsi extract dan mencetak respon dari API dengan pprint
stock_data, news_data = extract()
print("Response from Polygon.io API (Stock Data):")
pprint(stock_data)
print("\nResponse from Polygon.io API (News Data):")
pprint(news_data)

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

# Mengubah JSON menjadi DataFrame untuk Polygon.io News API dan memfilter berdasarkan simbol NVDA
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

# Proses data stock
df_stock = process_stock_data(stock_data)

# Proses data news dan filter berdasarkan simbol NVDA
df_news = process_news_data(news_data, 'NVDA')

print("\nDataFrame dari Polygon.io API (Stock Data):")
print(df_stock)

print("\nDataFrame dari Polygon.io API (News Data):")
print(df_news)

def transform(df_stock, df_news):
    # Download vader lexicon for sentiment analysis
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    # Extract necessary columns from news and create a copy
    if 'published_utc' in df_news.columns and 'title' in df_news.columns:
        news_df = df_news[['published_utc', 'title', 'sentiment', 'sentiment_reasoning']].copy()
        
        # Convert published_utc to datetime
        news_df['published_utc'] = pd.to_datetime(news_df['published_utc'])
        news_df['date'] = news_df['published_utc'].dt.date
        
        # Sentiment analysis for each headline
        news_df['sentiment_scores'] = news_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
        
        # Aggregate sentiment scores by date
        sentiment_agg_nltk = news_df.groupby('date')['sentiment_scores'].mean().reset_index()
        sentiment_agg_nltk.columns = ['date', 'average_sentiment_nltk']
        
        # Aggregate sentiment scores by date from insights
        sentiment_agg_insights = news_df.groupby('date')['sentiment'].apply(lambda x: x.mode()[0]).reset_index()
        sentiment_agg_insights.columns = ['date', 'average_sentiment_insights']
        
        # Merge both sentiment aggregations
        sentiment_agg = pd.merge(sentiment_agg_nltk, sentiment_agg_insights, on='date')
        
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
        print(sentiment_agg_nltk)
        
        print("\nAggregated Sentiment DataFrame before merge:")
        print(sentiment_agg)
        
        # Merge sentiment and stock data by date
        final_df = pd.merge(stock_df, sentiment_agg, on='date', how='inner')
        
        # Debugging: Print final DataFrame after merge
        print("\nFinal DataFrame after merge:")
        print(final_df)
        
        # Sort by date
        final_df = final_df.sort_values('date')
        
        # Arrange column order
        final_df = final_df[['date', 'close', 'average_sentiment_nltk', 'average_sentiment_insights']]
        
        return final_df, sentiment_agg, sentiment_agg_nltk
    
    else:
        print("Kolom yang diperlukan tidak ditemukan dalam data berita.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Example usage
final_df, sentiment_agg_df, sentiment_before_merge_df = transform(df_stock, df_news)
print("\nSentiment DataFrame before merge:")
print(sentiment_before_merge_df)
print("\nAggregated Sentiment DataFrame:")
print(sentiment_agg_df)
print("\nFinal DataFrame after merge:")
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