from src.Advanced_Stock_Price_Prediction_System import analyze_stock

# Analyze a stock
results = analyze_stock('AAPL')

# Print prediction
print(f"Prediction for {results['symbol']}:")
print(f"Direction: {results['prediction']['prediction']}")
print(f"Confidence: {results['prediction']['confidence']:.2f}")
print(f"Support/Resistance Levels: {results['prediction']['support_resistance']}")

# Charts are saved in the 'stock_analysis' directory
print(f"Analysis charts saved in: {results['charts_location']}")