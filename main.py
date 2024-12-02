from src.technical_analysis import AdvancedTechnicalAnalysis
from src.advanced_ml_model import AdvancedStockPredictor
import yfinance as yf

# Fetch data and analyze
symbol = "AAPL"
data = yf.download(symbol, start="2018-01-01")

# Technical Analysis
ta = AdvancedTechnicalAnalysis(data)
feature_data = ta.calculate_all_indicators()
signals = ta.get_signals()

# ML Prediction
predictor = AdvancedStockPredictor(feature_data)
# ... (rest of the prediction process)