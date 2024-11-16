import os
from dotenv import load_dotenv
import requests
from pprint import pprint
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd

def extract():
    load_dotenv()
    # Mengambil data dari News API
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY') 
    news_api_key = os.getenv('NEWS_API_KEY')    
    symbol = 'INTC'  # Simbol saham IBM
    q = 'intel'  # Query untuk pencarian berita
    url_stock = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={alpha_vantage_api_key}'
    url_news = f'https://newsapi.org/v2/everything?q={q}&from=2024-11-14&to=2024-11-14&sortBy=popularity&apiKey={news_api_key}'
    
    # Mengambil data dari API
    response_stock = requests.get(url_stock)
    data_stock = response_stock.json()

    response_news = requests.get(url_news)
    data_news = response_news.json()
    
    return data_stock, data_news

# Menjalankan fungsi extract dan mencetak respon dari API dengan pprint
data_stock, data_news = extract()
# print("Response from Alpha Vantage API (Stock Data):")
# pprint(data_stock)
# print("\nResponse from News API (News Data):")
# pprint(data_news)

# Mengubah JSON menjadi DataFrame untuk Alpha Vantage
def process_stock_data(data_stock):
    try:
        df_stock = pd.DataFrame.from_dict(data_stock['Time Series (5min)'], orient='index')
        df_stock.index = pd.to_datetime(df_stock.index)
        df_stock.columns = ['open', 'high', 'low', 'close', 'volume']
        return df_stock
    except KeyError:
        print("Kunci 'Time Series (5min)' tidak ditemukan dalam data JSON.")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong atau lakukan penanganan lain

# Mengubah JSON menjadi DataFrame untuk News API
df_news = pd.json_normalize(data_news['articles']) if 'articles' in data_news else pd.DataFrame()

# Proses data stock
df_stock = process_stock_data(data_stock)

# print("\nDataFrame dari Alpha Vantage API (Stock Data):")
# print(df_stock)

# print("\nDataFrame dari News API (News Data):")
# print(df_news)

# def transform(df_stock, df_news):
#     # Download vader lexicon untuk analisis sentimen
#     nltk.download('vader_lexicon')
#     sia = SentimentIntensityAnalyzer()
    
#     # Mengambil hanya kolom yang diperlukan dari news
#     news_df = df_news[['publishedAt', 'title']].copy()
    
#     # Konversi publishedAt ke datetime
#     news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    
#     # Analisis sentimen untuk setiap headline
#     news_df['sentiment_scores'] = news_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
#     # Agregasi sentimen per tanggal
#     daily_sentiment = news_df.groupby(news_df['publishedAt'].dt.date)['sentiment_scores'].mean().reset_index()
    
#     daily_sentiment.columns = ['date', 'avg_sentiment']
    
#     # Memproses data saham
#     stock_df = df_stock.reset_index()
#     stock_df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
#     stock_df['date'] = stock_df['datetime'].dt.date
    
#     # Mengambil hanya kolom yang diperlukan dari stock
#     stock_df = stock_df[['date', 'close']].copy()
    
#     # Menggabungkan data sentiment dan stock berdasarkan tanggal
#     final_df = pd.merge(stock_df, daily_sentiment, on='date', how='inner')
    
#     # Mengurutkan berdasarkan tanggal
#     final_df = final_df.sort_values('date')
    
#     return final_df
def transform(df_stock, df_news):
    # Download vader lexicon untuk analisis sentimen
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    # Mengambil kolom yang diperlukan dari news dan membuat copy
    news_df = df_news[['publishedAt', 'title']].copy()
    
    # Konversi publishedAt ke datetime
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    news_df['date'] = news_df['publishedAt'].dt.date
    
    # Analisis sentimen untuk setiap headline
    news_df['sentiment_scores'] = news_df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Memproses data saham
    stock_df = df_stock.reset_index()
    stock_df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    stock_df['date'] = stock_df['datetime'].dt.date
    
    # Mengambil kolom yang diperlukan dari stock
    stock_df = stock_df[['date', 'close']].copy()
    
    # Menggabungkan data sentiment dan stock berdasarkan tanggal
    final_df = pd.merge(stock_df, news_df[['date', 'title', 'sentiment_scores']], on='date', how='inner')
    
    # Mengurutkan berdasarkan tanggal
    final_df = final_df.sort_values('date')
    
    # Mengatur urutan kolom
    final_df = final_df[['date', 'close', 'title', 'sentiment_scores']]
    
    return final_df


# Contoh penggunaan
final_df = transform(df_stock, df_news)
print("\nDataFrame Hasil Gabungan:")
print(final_df)