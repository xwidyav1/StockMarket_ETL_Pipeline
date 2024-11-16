import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def fetch_data():
    try:
        # Konfigurasi koneksi database
        conn = psycopg2.connect(
            host='pg-353b3b2c-etl-stock-sentiment-analysis.h.aivencloud.com',
            port='20031',
            database='stock_market_db',
            user='avnadmin',
            password='AVNS_j2Ir2GgZqzJJ82yHL_s',
            sslmode='require'
        )
        
        # Membuat query - hanya mengambil close dan average_sentiment_nltk
        query = """
            SELECT date, close, average_sentiment_nltk
            FROM public.stock_sentiment 
            ORDER BY date
        """
        
        # Membaca data menggunakan pandas
        df = pd.read_sql(query, conn)
        
        # Tutup koneksi
        conn.close()
        
        return df
    
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def prepare_features(df):
    # Menggunakan average_sentiment_nltk sebagai feature tunggal
    X = df[['average_sentiment_nltk']]
    y = df['close']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_model(X, y):
    # Split data menjadi training dan testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Inisialisasi dan training model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Hitung metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, mse_train, mse_test, r2_train, r2_test

def plot_results(X_test, y_test, y_pred_test, model):
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Close Price')
    plt.ylabel('Predicted Close Price')
    plt.title('Actual vs Predicted Close Prices')
    
    # Plot 2: Sentiment vs Price with Regression Line
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
    
    # Creating regression line
    X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r--', label='Regression Line')
    
    plt.xlabel('NLTK Sentiment Score (Scaled)')
    plt.ylabel('Close Price')
    plt.title('Sentiment vs Price Relationship')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_data_quality(df):
    print("\nData Quality Analysis:")
    print("=====================")
    print("\n1. Data Shape:", df.shape)
    print("\n2. Missing Values:")
    print(df.isnull().sum())
    
    print("\n3. Basic Statistics:")
    print(df.describe())
    
    print("\n4. Correlation:")
    print(df[['close', 'average_sentiment_nltk']].corr())

def main():
    # Fetch data
    df = fetch_data()
    if df is None or df.empty:
        print("Failed to fetch data from database")
        return
    
    # Analyze data quality
    analyze_data_quality(df)
    
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model and get results
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, mse_train, mse_test, r2_train, r2_test = train_model(X, y)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Training MSE: {mse_train:.2f}")
    print(f"Testing MSE: {mse_test:.2f}")
    print(f"Training R² Score: {r2_train:.4f}")
    print(f"Testing R² Score: {r2_test:.4f}")
    print("\nModel Details:")
    print(f"Sentiment coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Plot results
    plot_results(X_test, y_test, y_pred_test, model)

if __name__ == "__main__":
    main()