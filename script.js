async function fetchStockData() {
    const response = await fetch('http://localhost:5000/fetch_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            ticker: 'AAPL',
            start_date: '2020-01-01',
            end_date: '2021-01-01',
            api_key: 'USLF3P1QRCK9TLCC'
        })
    });
    const data = await response.json();
    console.log(data);
}

async function trainModel() {
    const response = await fetch('http://localhost:5000/train_model', {
        method: 'POST'
    });
    const data = await response.json();
    console.log(data);
}

async function forecast() {
    const response = await fetch('http://localhost:5000/forecast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model_type: 'RandomForest',
            periods: 30
        })
    });
    const data = await response.json();
    console.log(data);
    plotForecast(data);
}

function plotForecast(data) {
    const dates = data.map(d => d.Date);
    const predictions = data.map(d => d.Predicted);
    const trace = {
        x: dates,
        y: predictions,
        type: 'scatter'
    };
    Plotly.newPlot('prediction-chart', [trace]);
}

async function optimizePortfolio() {
    const response = await fetch('http://localhost:5000/portfolio_optimize', {
        method: 'POST'
    });
    const data = await response.json();
    console.log(data);
}

fetchStockData();
trainModel();
forecast();
optimizePortfolio();
