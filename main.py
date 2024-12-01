from src.Advanced_Stock_Price_Prediction_System import analyze_stock

# Analyze a stock
results = analyze_stock('AAPL')

# Print prediction
print(f"Prediction for {results['symbol']}:")
print(f"Direction: {results['prediction']['prediction']}")
print(f"Confidence: {results['prediction']['confidence']:.2f}")
print(f"Current Price: ${results['prediction']['current_price']:.2f}")