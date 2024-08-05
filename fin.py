import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
import base64

# Initialize Dash app
app = dash.Dash(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['financial_data']

# Define functions to fetch, clean, and process data
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

def forecast_future_values(model, data, periods=30):
    data = data.ffill().bfill()
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    future_features = pd.DataFrame(index=future_dates, columns=data.columns).fillna(data.iloc[-1])
    if hasattr(model, 'feature_names_in_'):
        future_features = future_features[model.feature_names_in_]
    future_predictions = model.predict(future_features)
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
    return future_data

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Function to save data to MongoDB
def save_to_mongo(collection_name, data, db):
    collection = db[collection_name]
    collection.drop()  # Drop the collection if it exists
    data_dict = data.reset_index().to_dict(orient='records')
    collection.insert_many(data_dict)

# Dash app layout
app.layout = html.Div([
    html.H1("Financial Analysis Dashboard", style={'textAlign': 'center', 'color': '#0044cc'}),
    
    dcc.Input(id='ticker-input', type='text', placeholder='Enter Stock Ticker', style={'margin-right': '10px'}),
    dcc.Input(id='start-date-input', type='text', placeholder='YYYY-MM-DD', style={'margin-right': '10px'}),
    dcc.Input(id='end-date-input', type='text', placeholder='YYYY-MM-DD', style={'margin-right': '10px'}),
    dcc.Input(id='api-key-input', type='text', placeholder='Alpha Vantage API Key', style={'margin-right': '10px'}),
    dcc.Input(id='current-investment-input', type='number', placeholder='Current Investment Amount', style={'margin-right': '10px'}),
    dcc.Input(id='goal-amount-input', type='number', placeholder='Investment Goal Amount', style={'margin-right': '10px'}),
    
    html.Button('Submit', id='submit-button', n_clicks=0, style={'margin-bottom': '10px'}),
    
    html.Div(id='data-tables'),
    html.Div(id='graphs'),
    html.Div(id='recommendations'),
    html.Div(id='investment-analysis'),
    html.Div(id='download-links')
])

@app.callback(
    [Output('data-tables', 'children'),
     Output('graphs', 'children'),
     Output('recommendations', 'children'),
     Output('investment-analysis', 'children'),
     Output('download-links', 'children')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value'),
     dash.dependencies.State('start-date-input', 'value'),
     dash.dependencies.State('end-date-input', 'value'),
     dash.dependencies.State('api-key-input', 'value'),
     dash.dependencies.State('current-investment-input', 'value'),
     dash.dependencies.State('goal-amount-input', 'value')]
)
def update_dashboard(n_clicks, ticker, start_date, end_date, api_key, current_investment, goal_amount):
    if n_clicks > 0 and ticker and start_date and end_date and api_key and goal_amount is not None:
        # Fetch and process data
        yahoo_data = fetch_yahoo_finance_data(ticker, start_date, end_date)
        alpha_data = fetch_alpha_vantage_data(ticker, api_key)
        cleaned_yahoo_data = clean_data(yahoo_data)
        normalized_yahoo_data = normalize_data(cleaned_yahoo_data)
        standardized_yahoo_data = standardize_data(cleaned_yahoo_data)
        
        # Save to MongoDB
        save_to_mongo('yahoo_finance', cleaned_yahoo_data, db)
        save_to_mongo('alpha_vantage', alpha_data, db)
        save_to_mongo('normalized_yahoo_finance', normalized_yahoo_data, db)
        save_to_mongo('standardized_yahoo_finance', standardized_yahoo_data, db)
        
        # Prepare training data
        X = standardized_yahoo_data.drop('Close', axis=1)
        y = standardized_yahoo_data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        gb_model.fit(X_train, y_train)
        
        # Forecast future values
        rf_future_data = forecast_future_values(rf_model, cleaned_yahoo_data)
        gb_future_data = forecast_future_values(gb_model, cleaned_yahoo_data)
        
        # Generate plots
        rf_plot = plt.figure(figsize=(12, 6))
        plt.plot(rf_future_data['Date'], rf_future_data['Predicted'], label='Random Forest Predictions', color='orange')
        plt.title('Future Price Predictions - Random Forest')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True)
        rf_plot_base64 = plot_to_base64(rf_plot)
        
        gb_plot = plt.figure(figsize=(12, 6))
        plt.plot(gb_future_data['Date'], gb_future_data['Predicted'], label='Gradient Boosting Predictions', color='blue')
        plt.title('Future Price Predictions - Gradient Boosting')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True)
        gb_plot_base64 = plot_to_base64(gb_plot)
        
        # Calculate and display recommendations
        def evaluate_forecasted_prices(predictions, threshold=0.05):
            expected_return = np.mean(predictions)
            recommendation = "Buy" if expected_return > threshold else "Do not buy"
            return expected_return, recommendation
        
        rf_expected_return, rf_recommendation = evaluate_forecasted_prices(rf_future_data['Predicted'])
        gb_expected_return, gb_recommendation = evaluate_forecasted_prices(gb_future_data['Predicted'])
        
        def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
            excess_returns = returns - risk_free_rate
            return np.mean(excess_returns) / np.std(excess_returns)
        
        rf_sharpe_ratio = calculate_sharpe_ratio(rf_future_data['Predicted'])
        gb_sharpe_ratio = calculate_sharpe_ratio(gb_future_data['Predicted'])
        
        rf_recommendation_text = f"Random Forest: Expected Return: ${rf_expected_return:.2f}, Recommendation: {rf_recommendation}, Sharpe Ratio: {rf_sharpe_ratio:.2f}"
        gb_recommendation_text = f"Gradient Boosting: Expected Return: ${gb_expected_return:.2f}, Recommendation: {gb_recommendation}, Sharpe Ratio: {gb_sharpe_ratio:.2f}"
        
        # Calculate investment goal analysis
        def investment_goal_analysis(expected_return, goal_amount, current_investment):
            if expected_return <= 0:
                return 0
            required_investment = (goal_amount / expected_return) - current_investment
            return max(required_investment, 0)
        
        rf_required_investment = investment_goal_analysis(rf_expected_return, goal_amount, current_investment)
        gb_required_investment = investment_goal_analysis(gb_expected_return, goal_amount, current_investment)
        
        # Create investment analysis text
        investment_analysis = html.Div([
            html.H3("Investment Goal Analysis", style={'color': '#0044cc'}),
            html.P(f"Currently investing: ${current_investment:.2f}", style={'font-size': '16px', 'background-color': '#e8f5e9', 'padding': '15px', 'border-radius': '8px'}),
            html.P(f"To achieve your goal of ${goal_amount}, you should invest approximately ${rf_required_investment:.2f} using the Random Forest model.",
                   style={'font-size': '16px', 'background-color': '#e8f5e9', 'padding': '15px', 'border-radius': '8px'}),
            html.P(f"To achieve your goal of ${goal_amount}, you should invest approximately ${gb_required_investment:.2f} using the Gradient Boosting model.",
                   style={'font-size': '16px', 'background-color': '#e8f5e9', 'padding': '15px', 'border-radius': '8px'})
        ])
        
        # Provide download links
        download_links = html.Div([
            html.A("Download Normalized Data CSV", href="/download/normalized_data.csv", style={'display': 'block', 'margin-bottom': '10px'}),
            html.A("Download Standardized Data CSV", href="/download/standardized_data.csv", style={'display': 'block', 'margin-bottom': '10px'}),
            html.A("Download Future Predictions CSV", href="/download/future_predictions.csv", style={'display': 'block', 'margin-bottom': '10px'}),
        ])
        
        # Return results
        return (
            html.Div([
                html.H3("Cleaned Data", style={'color': '#0044cc'}),
                dcc.Graph(
                    figure={
                        'data': [dict(x=cleaned_yahoo_data.index, y=cleaned_yahoo_data['Close'], type='line', name='Close Price')],
                        'layout': {
                            'title': 'Cleaned Yahoo Finance Data',
                            'xaxis': {'title': 'Date'},
                            'yaxis': {'title': 'Price'},
                            'plot_bgcolor': '#f9f9f9'
                        }
                    }
                )
            ]),
            html.Div([
                html.Img(src='data:image/png;base64,' + rf_plot_base64, style={'display': 'block', 'margin-bottom': '10px'}),
                html.Img(src='data:image/png;base64,' + gb_plot_base64, style={'display': 'block', 'margin-bottom': '10px'}),
            ]),
            html.Div([
                html.H3("Recommendations", style={'color': '#0044cc'}),
                html.P(rf_recommendation_text, style={'font-size': '16px'}),
                html.P(gb_recommendation_text, style={'font-size': '16px'}),
            ], style={'background-color': '#e8f5e9', 'padding': '15px', 'border-radius': '8px'}),
            investment_analysis,
            download_links
        )
    else:
        # Return empty components if no input is provided
        return (html.Div([]), html.Div([]), html.Div([]), html.Div([]), html.Div([]))

if __name__ == '__main__':
    app.run_server(debug=True)
