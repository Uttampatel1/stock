import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import datetime

def analyze_stock(symbol: str):
    print(f"\nAnalyzing {symbol}...")
    
    # 1. Fetch data
    print("Fetching data...")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    df = yf.download(symbol, start=start_date, end=end_date)
    print("Data fetched successfully")
    
    # 2. Create features
    print("Creating features...")
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Create target: 1 if tomorrow's price is higher, 0 if lower
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Remove any rows with NaN values
    df.dropna(inplace=True)
    
    # 3. Prepare features and target
    feature_columns = ['Returns', 'SMA_20', 'SMA_50', 'Daily_Range', 'Volume_Change']
    X = df[feature_columns][:-1]  # Remove last row as we don't have next day's price
    y = df['Target'][:-1]  # Remove last row as we don't have next day's price
    
    # 4. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert target to 1D array
    y = y.values
    
    # 5. Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # 6. Prepare prediction for tomorrow
    latest_features = df[feature_columns].iloc[[-1]]
    latest_features_scaled = scaler.transform(latest_features)
    
    # Make prediction
    prediction = model.predict(latest_features_scaled)[0]
    probability = model.predict_proba(latest_features_scaled)[0]
    confidence = probability[1] if prediction == 1 else probability[0]
    
    # Get current values
    current_price = float(df['Close'].iloc[-1])
    sma_20 = float(df['SMA_20'].iloc[-1])
    sma_50 = float(df['SMA_50'].iloc[-1])
    daily_range = float(df['Daily_Range'].iloc[-1])
    volume_change = float(df['Volume_Change'].iloc[-1])
    
    # 7. Print results
    print("\nAnalysis Results:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Prediction for tomorrow: {'HIGHER' if prediction == 1 else 'LOWER'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nCurrent Technical Indicators:")
    print(f"20-day SMA: ${sma_20:.2f}")
    print(f"50-day SMA: ${sma_50:.2f}")
    print(f"Today's Trading Range: {daily_range:.2%}")
    print(f"Volume Change: {volume_change:.2%}")
    
    # 8. Save results to file
    results = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'symbol': symbol,
        'current_price': current_price,
        'prediction': 'HIGHER' if prediction == 1 else 'LOWER',
        'confidence': float(confidence),
        'technical_indicators': {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'daily_range': daily_range,
            'volume_change': volume_change
        }
    }
    
    # Save to JSON file
    filename = f'{symbol}_prediction_{datetime.datetime.now().strftime("%Y%m%d")}.json'
    with open(filename, 'w') as f:
        import json
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_stock("AAPL")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        