from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pymongo import MongoClient
import numpy as np
from scipy.optimize import minimize
import json

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['financial_data']

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def fetch_alpha_vantage_data(ticker, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    return data

def clean_data(data):
    data = data.drop_duplicates()
    data = data.fillna(method='ffill').fillna(method='bfill')
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return normalized_data

def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return standardized_data

def save_to_mongo(collection_name, data, db):
    collection = db[collection_name]
    collection.delete_many({})
    collection.insert_many(data.reset_index().to_dict('records'))

@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    req_data = request.get_json()
    ticker = req_data['ticker']
    start_date = req_data['start_date']
    end_date = req_data['end_date']
    api_key = req_data['api_key']

    yahoo_data = fetch_yahoo_finance_data(ticker, start_date, end_date)
    alpha_data = fetch_alpha_vantage_data(ticker, api_key)
    
    cleaned_yahoo_data = clean_data(yahoo_data)
    cleaned_alpha_data = clean_data(alpha_data)
    
    normalized_yahoo_data = normalize_data(cleaned_yahoo_data)
    standardized_yahoo_data = standardize_data(cleaned_yahoo_data)
    
    save_to_mongo('yahoo_finance', cleaned_yahoo_data, db)
    save_to_mongo('alpha_vantage', cleaned_alpha_data, db)
    save_to_mongo('normalized_yahoo_finance', normalized_yahoo_data, db)
    save_to_mongo('standardized_yahoo_finance', standardized_yahoo_data, db)
    
    return jsonify({"status": "Data fetched and saved successfully"}), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    collection = db['standardized_yahoo_finance']
    data = pd.DataFrame(list(collection.find()))
    data.set_index('Date', inplace=True)
    
    X = data.drop('Close', axis=1)
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    
    rf_predictions = rf_model.predict(X_test)
    gb_predictions = gb_model.predict(X_test)
    
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    gb_r2 = r2_score(y_test, gb_predictions)
    
    return jsonify({
        "Random Forest": {"MSE": rf_mse, "R2": rf_r2},
        "Gradient Boosting": {"MSE": gb_mse, "R2": gb_r2}
    }), 200

@app.route('/forecast', methods=['POST'])
def forecast():
    req_data = request.get_json()
    model_type = req_data['model_type']
    periods = req_data['periods']

    collection = db['standardized_yahoo_finance']
    data = pd.DataFrame(list(collection.find()))
    data.set_index('Date', inplace=True)

    X = data.drop('Close', axis=1)
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    model.fit(X_train, y_train)

    # Ensure future_data has the same columns as X
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods)
    future_features = pd.DataFrame(index=future_dates, columns=X.columns)
    future_features.fillna(method='ffill', inplace=True)

    future_predictions = model.predict(future_features)
    
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
    
    return future_data.to_json(orient='records'), 200

@app.route('/portfolio_optimize', methods=['POST'])
def portfolio_optimize():
    collection = db['yahoo_finance']
    data = pd.DataFrame(list(collection.find()))
    data.set_index('Date', inplace=True)
    
    historical_returns = data.pct_change().dropna()
    
    def portfolio_performance(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return portfolio_return, portfolio_risk

    def minimize_sharpe_ratio(weights, returns, risk_free_rate=0.01):
        portfolio_return, portfolio_risk = portfolio_performance(weights, returns)
        return - (portfolio_return - risk_free_rate) / portfolio_risk

    num_assets = len(historical_returns.columns)
    initial_weights = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    result = minimize(minimize_sharpe_ratio, initial_weights, args=(historical_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    
    return jsonify({"Optimal Weights": optimal_weights.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)
