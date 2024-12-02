# main.py (continued)
def main():
    """Main function to run the stock prediction system"""
    try:
        import yfinance as yf
        from datetime import timedelta
        # from datetime import datetime, timedelta
        
        # Get stock symbol from user
        symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Setup logging to file
        log_file = output_dir / f"stock_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Starting prediction process for {symbol}")
        
        # Download historical data
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        logger.info(f"Downloaded {len(data)} days of historical data")
        
        # Calculate technical indicators
        ta = AdvancedTechnicalAnalysis(data)
        feature_data = ta.calculate_all_indicators()
        logger.info("Calculated technical indicators")
        
        # Initialize predictor
        predictor = AdvancedStockPredictor(feature_data)
        
        # Prepare data
        logger.info("Preparing data for training...")
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close')
        
        # Optimize hyperparameters
        logger.info("Optimizing hyperparameters...")
        best_params = predictor.optimize_hyperparameters(X_train, y_train)
        logger.info(f"Best parameters: {best_params}")
        
        # Train models
        logger.info("Training ensemble models...")
        results = predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
        # Calculate feature importance
        importance = predictor.calculate_feature_importance(feature_names)
        
        # Make prediction for next day
        latest_data = feature_data.iloc[-1:].drop('Close', axis=1)
        prediction = predictor.predict_next_day(latest_data)
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"{symbol}_model_{timestamp}.joblib"
        predictor.save_model(model_path)
        
        # Save feature importance plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        importance['mean_importance'].plot(kind='bar')
        plt.title(f'Feature Importance for {symbol}')
        plt.tight_layout()
        plt.savefig(output_dir / f"{symbol}_feature_importance_{timestamp}.png")
        
        # Print results
        print(f"\nPrediction Results for {symbol}:")
        print(f"Ensemble Prediction: {prediction['ensemble_prediction']}")
        print(f"Confidence: {prediction['ensemble_probability']:.2%}")
        
        print("\nModel Performance:")
        for model, metrics in results.items():
            print(f"\n{model.upper()}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        print("\nTop 10 Most Important Features:")
        print(importance['mean_importance'].head(10))
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()


# Example usage for backtesting
def backtest_strategy(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Backtest the prediction strategy
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with backtesting results
    """
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Calculate indicators
        ta = AdvancedTechnicalAnalysis(data)
        feature_data = ta.calculate_all_indicators()
        
        # Initialize predictor
        predictor = AdvancedStockPredictor(feature_data)
        
        # Prepare data with smaller test size for more backtesting data
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data('Close', test_size=0.3)
        
        # Train models
        best_params = predictor.optimize_hyperparameters(X_train, y_train, n_trials=30)
        predictor.train_ensemble(X_train, y_train, X_test, y_test, best_params)
        
        # Create backtest DataFrame
        backtest_data = data[int(len(data) * 0.7):].copy()  # Use test portion
        predictions = []
        probabilities = []
        
        # Make predictions for each day
        for i in range(len(backtest_data) - 1):
            current_features = feature_data.iloc[i:i+1].drop('Close', axis=1)
            pred = predictor.predict_next_day(current_features)
            predictions.append(1 if pred['ensemble_prediction'] == 'INCREASE' else 0)
            probabilities.append(pred['ensemble_probability'])
        
        backtest_data = backtest_data[:-1]  # Remove last day since we don't have next day's price
        backtest_data['Predicted_Direction'] = predictions
        backtest_data['Prediction_Probability'] = probabilities
        backtest_data['Actual_Direction'] = (backtest_data['Close'].shift(-1) > backtest_data['Close']).astype(int)
        backtest_data['Correct_Prediction'] = (backtest_data['Predicted_Direction'] == backtest_data['Actual_Direction']).astype(int)
        
        # Calculate returns
        backtest_data['Strategy_Return'] = backtest_data['Correct_Prediction'] * abs(backtest_data['Close'].pct_change())
        backtest_data['Buy_Hold_Return'] = backtest_data['Close'].pct_change()
        
        return backtest_data
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise

# Example of running backtest
def run_backtest_example():
    """Run a backtest example"""
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    results = backtest_strategy(symbol, start_date, end_date)
    
    # Calculate performance metrics
    strategy_return = results['Strategy_Return'].sum()
    buy_hold_return = results['Buy_Hold_Return'].sum()
    accuracy = results['Correct_Prediction'].mean()
    
    print(f"\nBacktest Results for {symbol}:")
    print(f"Strategy Return: {strategy_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Prediction Accuracy: {accuracy:.2%}")
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    ((1 + results['Strategy_Return']).cumprod()).plot(label='Strategy')
    ((1 + results['Buy_Hold_Return']).cumprod()).plot(label='Buy & Hold')
    plt.title(f'Cumulative Returns: Strategy vs Buy & Hold ({symbol})')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run either the main prediction system or backtest
    choice = input("Enter 1 for prediction or 2 for backtest: ")
    if choice == "1":
        main()
    elif choice == "2":
        run_backtest_example()
    else:
        print("Invalid choice")