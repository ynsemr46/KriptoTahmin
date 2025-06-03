import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
    has_prophet = True
except ImportError:
    has_prophet = False
import plotly.graph_objs as go
import plotly.io as pio
from flask import Flask, render_template_string, request
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    has_xgb = True
except ImportError:
    has_xgb = False
try:
    import lightgbm as lgb
    has_lgb = True
except ImportError:
    has_lgb = False
try:
    import torch
    import torch.nn as nn
    has_torch = True
except ImportError:
    has_torch = False
import aiohttp
import asyncio
import sqlite3
import optuna
import json

app = Flask(__name__)

COINS = {
    "bitcoin": {"name": "Bitcoin", "symbol": "BTC"},
    "ethereum": {"name": "Ethereum", "symbol": "ETH"},
    "solana": {"name": "Solana", "symbol": "SOL"},
    "binancecoin": {"name": "Binance Coin", "symbol": "BNB"},
    "ripple": {"name": "XRP", "symbol": "XRP"},
    "cardano": {"name": "Cardano", "symbol": "ADA"},
    "dogecoin": {"name": "Dogecoin", "symbol": "DOGE"},
    "tron": {"name": "TRON", "symbol": "TRX"},
    "polkadot": {"name": "Polkadot", "symbol": "DOT"},
    "avalanche-2": {"name": "Avalanche", "symbol": "AVAX"},
    "chainlink": {"name": "Chainlink", "symbol": "LINK"},
    "litecoin": {"name": "Litecoin", "symbol": "LTC"},
    "matic-network": {"name": "Polygon", "symbol": "MATIC"},
    "uniswap": {"name": "Uniswap", "symbol": "UNI"},
    "stellar": {"name": "Stellar", "symbol": "XLM"},
    "bitcoin-cash": {"name": "Bitcoin Cash", "symbol": "BCH"},
    "internet-computer": {"name": "Internet Computer", "symbol": "ICP"},
    "vechain": {"name": "VeChain", "symbol": "VET"},
    "aptos": {"name": "Aptos", "symbol": "APT"},
    "arbitrum": {"name": "Arbitrum", "symbol": "ARB"},
    "pepe": {"name": "Pepe", "symbol": "PEPE"},
    "dogwifhat": {"name": "dogwifhat", "symbol": "WIF"},
    "toncoin": {"name": "Toncoin", "symbol": "TON"},
    "shiba-inu": {"name": "Shiba Inu", "symbol": "SHIB"},
    "render-token": {"name": "Render", "symbol": "RNDR"},
    "injective-protocol": {"name": "Injective", "symbol": "INJ"},
    "optimism": {"name": "Optimism", "symbol": "OP"},
    "the-graph": {"name": "The Graph", "symbol": "GRT"},
    "maker": {"name": "Maker", "symbol": "MKR"},
    "kaspa": {"name": "Kaspa", "symbol": "KAS"},
    "monero": {"name": "Monero", "symbol": "XMR"},
    "aave": {"name": "Aave", "symbol": "AAVE"},
    "sui": {"name": "Sui", "symbol": "SUI"},
    "mina-protocol": {"name": "Mina", "symbol": "MINA"},
    "osmosis": {"name": "Osmosis", "symbol": "OSMO"},
    "fantom": {"name": "Fantom", "symbol": "FTM"},
    "thorchain": {"name": "THORChain", "symbol": "RUNE"},
    "arweave": {"name": "Arweave", "symbol": "AR"},
    "gala": {"name": "GALA", "symbol": "GALA"},
    "flow": {"name": "Flow", "symbol": "FLOW"},
    "bitcoin-sv": {"name": "Bitcoin SV", "symbol": "BSV"},
    "tezos": {"name": "Tezos", "symbol": "XTZ"},
    "algorand": {"name": "Algorand", "symbol": "ALGO"},
    "decentraland": {"name": "Decentraland", "symbol": "MANA"},
    "axie-infinity": {"name": "Axie Infinity", "symbol": "AXS"},
    "enjincoin": {"name": "Enjin Coin", "symbol": "ENJ"},
    "basic-attention-token": {"name": "Basic Attention Token", "symbol": "BAT"},
    "iota": {"name": "IOTA", "symbol": "IOTA"},
    "zcash": {"name": "Zcash", "symbol": "ZEC"},
    "dash": {"name": "Dash", "symbol": "DASH"},
    "waves": {"name": "Waves", "symbol": "WAVES"},
    "chiliz": {"name": "Chiliz", "symbol": "CHZ"},
    "curve-dao-token": {"name": "Curve DAO", "symbol": "CRV"},
    "pancakeswap-token": {"name": "PancakeSwap", "symbol": "CAKE"},
    "1inch": {"name": "1inch", "symbol": "1INCH"},
    "convex-finance": {"name": "Convex Finance", "symbol": "CVX"},
    "compound-governance-token": {"name": "Compound", "symbol": "COMP"},
    "ankr": {"name": "Ankr", "symbol": "ANKR"},
    "terra-luna": {"name": "Terra Luna", "symbol": "LUNA"},
    "terra-luna-2": {"name": "Terra Luna 2.0", "symbol": "LUNA2"},
    "yearn-finance": {"name": "Yearn Finance", "symbol": "YFI"},
    "usd-coin": {"name": "USD Coin", "symbol": "USDC"},
    "true-usd": {"name": "TrueUSD", "symbol": "TUSD"},
    "dai": {"name": "Dai", "symbol": "DAI"},
    "frax": {"name": "Frax", "symbol": "FRAX"},
    "tether": {"name": "Tether", "symbol": "USDT"},
    "binance-usd": {"name": "Binance USD", "symbol": "BUSD"},
    "usd-digital-dollar": {"name": "Digital Dollar", "symbol": "DUSD"},
    "usdd": {"name": "USDD", "symbol": "USDD"},
    "usdp": {"name": "Pax Dollar", "symbol": "USDP"},
    "soil": {"name": "Soil", "symbol": "SOIL"},
}

# Veri önbellekleme
def cache_data(coin_id, df):
    conn = sqlite3.connect('cache.db')
    df.to_sql(coin_id, conn, if_exists='replace', index=True)
    conn.close()

def get_cached_data(coin_id):
    conn = sqlite3.connect('cache.db')
    try:
        df = pd.read_sql(f"SELECT * FROM {coin_id}", conn, index_col='date', parse_dates=['date'])
        return df
    except:
        return None
    finally:
        conn.close()

async def get_coin_data(coin_id):
    cached = get_cached_data(coin_id)
    if cached is not None and not cached.empty:
        return cached
    async with aiohttp.ClientSession() as session:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": "180", "interval": "daily"}
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if 'prices' not in data or not data['prices']:
                raise ValueError("Veri alınamadı veya coin id hatalı.")
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df = df[['price']]
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.asfreq('D')
            df['price'] = df['price'].interpolate(method='linear')
            mean = df['price'].mean()
            std = df['price'].std()
            df['price'] = np.clip(df['price'], mean - 3*std, mean + 3*std)
            df['log_price'] = np.log(df['price'] + 1e-6)  # Sıfır log hatasını önlemek için
            cache_data(coin_id, df)
            return df
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                await asyncio.sleep(60)  # Rate limit için bekle
                return await get_coin_data(coin_id)
            raise ValueError(f"API hatası: {str(e)}")

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_technical_indicators(df):
    df['SMA_7'] = df['price'].rolling(window=7).mean()
    df['EMA_14'] = df['price'].ewm(span=14, adjust=False).mean()
    df['RSI'] = compute_rsi(df['price'], 14)
    df['MACD'] = df['price'].ewm(span=12).mean() - df['price'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['ATR'] = df['price'].rolling(window=14).std()
    return df.dropna()

def create_features(df, lags=14, target_col='price'):
    df = create_technical_indicators(df)
    df_feat = df.copy()
    for i in range(1, lags+1):
        df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['day'] = df_feat.index.day
    df_feat = df_feat.dropna()
    return df_feat

def sarima_forecast(df, steps=30):
    try:
        model = SARIMAX(df['log_price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=steps)
        pred_uc_log = forecast.predicted_mean
        pred_ci_log = forecast.conf_int()
        pred_uc = np.exp(pred_uc_log)
        pred_ci = np.exp(pred_ci_log)
        return pred_uc, pred_ci
    except Exception:
        last_price = df['price'].iloc[-1]
        index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        pred_uc = pd.Series([last_price] * steps, index=index)
        pred_ci = pd.DataFrame({'lower': pred_uc * 0.98, 'upper': pred_uc * 1.02}, index=index)
        return pred_uc, pred_ci

def prophet_forecast(df, steps=30):
    if not has_prophet:
        raise ImportError("Prophet yüklü değil.")
    df = create_technical_indicators(df)
    pdf = df.reset_index()[['date', 'price', 'SMA_7', 'RSI']].rename(columns={'date': 'ds', 'price': 'y'})
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.add_regressor('SMA_7')
    m.add_regressor('RSI')
    m.fit(pdf)
    future = m.make_future_dataframe(periods=steps)
    future['SMA_7'] = df['SMA_7'].iloc[-1]  # Basit yaklaşım, daha iyi tahmin için geliştirilebilir
    future['RSI'] = df['RSI'].iloc[-1]
    forecast = m.predict(future)
    pred_uc = forecast['yhat'][-steps:].values
    pred_ci = np.vstack([forecast['yhat_lower'][-steps:], forecast['yhat_upper'][-steps:]]).T
    index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    return pd.Series(pred_uc, index=index), pd.DataFrame(pred_ci, index=index, columns=['lower', 'upper'])

def optimize_xgboost(df):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
        }
        df_feat = create_features(df, lags=14)
        X, y = df_feat.drop('price', axis=1), df_feat['price']
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        return -model.score(X, y)
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
    return study.best_params

def xgboost_forecast(df, steps=30):
    if not has_xgb:
        raise ImportError("XGBoost yüklü değil.")
    lags = 14
    df_feat = create_features(df, lags)
    X = df_feat.drop('price', axis=1).values
    y = df_feat['price'].values
    params = optimize_xgboost(df)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    last_row = df_feat.iloc[-1].copy()
    preds = []
    for _ in range(steps):
        features = last_row.drop('price').values.reshape(1, -1)
        pred = model.predict(features)[0]
        preds.append(pred)
        for i in range(lags, 1, -1):
            last_row[f'lag_{i}'] = last_row[f'lag_{i-1}']
        last_row['lag_1'] = pred
        next_date = last_row.name + pd.Timedelta(days=1)
        last_row['dayofweek'] = next_date.dayofweek
        last_row['month'] = next_date.month
        last_row['day'] = next_date.day
        last_row.name = next_date
    index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    pred_uc = pd.Series(preds, index=index)
    pred_ci = pd.DataFrame({'lower': pred_uc * 0.97, 'upper': pred_uc * 1.03}, index=index)
    return pred_uc, pred_ci

def lstm_forecast(df, steps=30):
    if not has_torch:
        raise ImportError("PyTorch yüklü değil.")
    lags = 14
    data = df['price'].values.astype(np.float32)
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(data[i])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    class AdvancedLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    model = AdvancedLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = nn.MSELoss()
    model.train()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(50):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
    model.eval()
    last_seq = torch.tensor(data[-lags:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            pred = model(last_seq).item()
            preds.append(pred)
            last_seq = torch.cat([last_seq[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)], dim=1)
    index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    pred_uc = pd.Series(preds, index=index)
    pred_ci = pd.DataFrame({'lower': pred_uc * 0.95, 'upper': pred_uc * 1.05}, index=index)
    return pred_uc, pred_ci

def sarima_xgboost_hybrid(df, steps=30):
    if not has_xgb:
        raise ImportError("XGBoost yüklü değil.")
    pred_uc_sarima, _ = sarima_forecast(df, steps)
    df['sarima_pred'] = np.exp(SARIMAX(df['log_price'], order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False).fittedvalues)
    df['resid'] = df['price'] - df['sarima_pred']
    lags = 7
    df_feat = create_features(df[['resid']], lags, target_col='resid')
    X = df_feat.drop('resid', axis=1).values
    y = df_feat['resid'].values
    params = optimize_xgboost(df[['resid']].rename(columns={'resid': 'price'}))
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    last_row = df_feat.iloc[-1].copy()
    preds = []
    for _ in range(steps):
        features = last_row.drop('resid').values.reshape(1, -1)
        pred = model.predict(features)[0]
        preds.append(pred)
        for i in range(lags, 1, -1):
            last_row[f'lag_{i}'] = last_row[f'lag_{i-1}']
        last_row['lag_1'] = pred
        next_date = last_row.name + pd.Timedelta(days=1)
        last_row['dayofweek'] = next_date.dayofweek
        last_row['month'] = next_date.month
        last_row['day'] = next_date.day
        last_row.name = next_date
    hybrid_pred = pred_uc_sarima.values + np.array(preds)
    index = pred_uc_sarima.index
    pred_uc = pd.Series(hybrid_pred, index=index)
    pred_ci = pd.DataFrame({'lower': pred_uc * 0.97, 'upper': pred_uc * 1.03}, index=index)
    return pred_uc, pred_ci

def weighted_ensemble_forecast(df, steps=30):
    preds = []
    weights = {'sarima': 0.3, 'prophet': 0.3, 'xgboost': 0.2, 'lstm': 0.2}
    model_names = []
    if has_prophet:
        try:
            prophet_pred, _ = prophet_forecast(df, steps)
            preds.append(prophet_pred.values * weights['prophet'])
            model_names.append("Prophet")
        except:
            pass
    if has_xgb:
        try:
            xgb_pred, _ = xgboost_forecast(df, steps)
            preds.append(xgb_pred.values * weights['xgboost'])
            model_names.append("XGBoost")
        except:
            pass
    try:
        sarima_pred, _ = sarima_forecast(df, steps)
        preds.append(sarima_pred.values * weights['sarima'])
        model_names.append("SARIMA")
    except:
        pass
    if has_torch:
        try:
            lstm_pred, _ = lstm_forecast(df, steps)
            preds.append(lstm_pred.values * weights['lstm'])
            model_names.append("LSTM")
        except:
            pass
    if not preds:
        raise ValueError("Hiçbir model çalışmadı.")
    ensemble_pred = np.sum(preds, axis=0)
    index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    pred_uc = pd.Series(ensemble_pred, index=index)
    pred_ci = pd.DataFrame({'lower': pred_uc * 0.95, 'upper': pred_uc * 1.05}, index=index)
    return pred_uc, pred_ci, model_names

def compare_models(df, steps=30):
    results = {}
    model_names = []
    colors = {'sarima': '#f2a365', 'prophet': '#00F2FE', 'xgboost': '#7209B7', 'lstm': '#F72585', 'ensemble': '#B8B8FF'}
    for model in ['sarima', 'prophet', 'xgboost', 'lstm', 'ensemble']:
        if (model == 'prophet' and not has_prophet) or (model == 'xgboost' and not has_xgb) or (model == 'lstm' and not has_torch):
            continue
        try:
            pred, _ = globals()[f"{model}_forecast"](df, steps)
            results[model] = pred
            model_names.append(model.upper())
        except:
            pass
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Gerçek Fiyat', line=dict(color='#22223b')))
    for model, pred in results.items():
        fig.add_trace(go.Scatter(x=pred.index, y=pred, mode='lines', name=model.upper(), line=dict(color=colors.get(model, '#000000'))))
    fig.update_layout(
        title='Model Karşılaştırması',
        xaxis_title='Tarih', yaxis_title='Fiyat (USD)',
        template='plotly_white', legend=dict(font=dict(size=14)),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn'), model_names

@app.route('/', methods=['GET'])
def index():
    coin_id = request.args.get('coin', 'bitcoin')
    model_type = request.args.get('model', 'sarima')
    days = int(request.args.get('days', 30))
    view = request.args.get('view', 'single')  # 'single' veya 'compare'
    coin = COINS.get(coin_id, COINS['bitcoin'])
    
    # Asenkron veri çekme
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(get_coin_data(coin_id))
    except Exception as e:
        return f"""
        <div style='background:#18122B;height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;'>
            <h2 style='color:#F72585;text-align:center;margin-bottom:24px;'>Veri alınamadı: {str(e)}</h2>
            <a href='/' style='padding:12px 28px;background:#7209B7;color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:1.1rem;transition:background 0.2s;'>Geri Dön</a>
        </div>
        """, 500
    finally:
        loop.close()

    # Model seçimi
    if view == 'compare':
        graph_html, used_models = compare_models(df, steps=days)
        model_name = "Model Karşılaştırması"
        pred_uc = None
        pred_ci = None
    else:
        if model_type == 'ensemble':
            pred_uc, pred_ci, used_models = weighted_ensemble_forecast(df, steps=days)
            model_name = "Ensemble (" + ", ".join(used_models) + ")"
        elif model_type == 'prophet' and has_prophet:
            pred_uc, pred_ci = prophet_forecast(df, steps=days)
            model_name = "Prophet+Exog"
        elif model_type == 'xgboost' and has_xgb:
            pred_uc, pred_ci = xgboost_forecast(df, steps=days)
            model_name = "XGBoost"
        elif model_type == 'lstm' and has_torch:
            pred_uc, pred_ci = lstm_forecast(df, steps=days)
            model_name = "LSTM"
        else:
            pred_uc, pred_ci = sarima_forecast(df, steps=days)
            model_name = "SARIMA"
        # --- Güven aralığı DataFrame'inde 'upper' ve 'lower' yoksa fallback ---
        if not isinstance(pred_ci, pd.DataFrame) or 'upper' not in pred_ci.columns or 'lower' not in pred_ci.columns:
            # Fallback: ±%2 bandı
            pred_ci = pd.DataFrame({'lower': pred_uc * 0.98, 'upper': pred_uc * 1.02}, index=pred_uc.index)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Gerçek Fiyat', line=dict(color='#22223b')))
        fig.add_trace(go.Scatter(x=pred_uc.index, y=pred_uc, mode='lines', name=f'{days} Günlük Tahmin', line=dict(color='#f2a365')))
        fig.add_trace(go.Scatter(
            x=pred_uc.index.tolist() + pred_uc.index[::-1].tolist(),
            y=pred_ci['upper'].tolist() + pred_ci['lower'][::-1].tolist(),
            fill='toself', fillcolor='rgba(242,163,101,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        ))
        fig.update_layout(
            title=f'{coin["name"]} Fiyat Tahmini ({model_name})',
            xaxis_title='Tarih', yaxis_title='Fiyat (USD)',
            template='plotly_white', legend=dict(font=dict(size=14)),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        graph_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # Tablo
    last_30 = df.tail(30).copy()
    last_30['type'] = 'Gerçek'
    last_30 = last_30.reset_index()
    if view != 'compare':
        forecast_df = pd.DataFrame({'price': pred_uc, 'type': 'Tahmin'})
        forecast_df['date'] = pred_uc.index
        forecast_df = forecast_df[['date', 'price', 'type']]
        table_df = pd.concat([last_30, forecast_df], ignore_index=True)
    else:
        table_df = last_30[['date', 'price', 'type']]
    table_df['date'] = pd.to_datetime(table_df['date']).dt.strftime('%d.%m.%Y')
    table_html = table_df.to_json(orient='records')
    spark_data = df['price'].tail(30).tolist() + (pred_uc.tolist() if view != 'compare' else [])

    html = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{coin_name}} {{days}} Günlük Fiyat Tahmini</title>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
        <style>
            body {
                background: #18122B;
                font-family: 'Segoe UI', 'Roboto', 'Montserrat', Arial, sans-serif;
                color: #F7F7F7;
                margin: 0;
            }
            .container {
                max-width: 900px;
                margin: 40px auto;
                background: #231942cc;
                border-radius: 18px;
                box-shadow: 0 4px 32px #5A189A44;
                padding: 36px 32px 32px 32px;
                backdrop-filter: blur(2px);
                animation: fadein 1.2s;
            }
            @media (max-width: 600px) {
                .container { padding: 15px; }
                h1 { font-size: 1.8rem; }
                select { width: 100%; margin: 10px 0; }
                .model-select { margin-left: 0; }
            }
            @keyframes fadein {
                from { opacity: 0; transform: translateY(40px); }
                to { opacity: 1; transform: none; }
            }
            h1 {
                font-size: 2.2rem;
                margin-bottom: 1.2rem;
                font-weight: 800;
                letter-spacing: 1px;
                color: #F72585;
            }
            label, select, .footer {
                color: #F7F7F7;
                font-size: 1.05rem;
            }
            select {
                background: #18122B;
                color: #F7F7F7;
                border: 1px solid #7209B7;
                border-radius: 8px;
                padding: 6px 12px;
                margin: 0 8px;
                font-weight: 600;
                outline: none;
                transition: border 0.2s;
            }
            select:focus {
                border: 1.5px solid #F72585;
            }
            .footer {
                margin-top: 2.5rem;
                font-size: 1rem;
                color: #B8B8FF;
                text-align: center;
                letter-spacing: 0.5px;
            }
            .sparkline {
                height: 32px;
                width: 140px;
                background: transparent;
            }
            .select-coin {
                margin-bottom: 28px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .model-select {
                margin-left: 18px;
            }
            #main-graph {
                margin-bottom: 18px;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 0 16px #7209B733;
                background: #18122B;
                animation: fadein 1.2s;
            }
            table.dataTable {
                background: #231942;
                color: #F7F7F7;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 0 8px #7209B744;
                animation: fadein 1.2s;
            }
            table.dataTable thead {
                background: #7209B7;
                color: #F7F7F7;
                font-weight: 700;
            }
            table.dataTable tbody tr {
                font-size: 1.08rem;
                background: #231942;
                color: #F7F7F7;
                border-bottom: 1px solid #7209B755;
                transition: background 0.2s;
            }
            table.dataTable tbody tr:hover {
                background: #7209B7cc;
                color: #F72585;
            }
            table.dataTable td, table.dataTable th {
                border: none;
            }
            .dataTables_wrapper .dataTables_paginate .paginate_button {
                color: #F72585 !important;
                background: #231942 !important;
                border: 1px solid #7209B7 !important;
                border-radius: 6px;
                margin: 0 2px;
            }
            .dataTables_wrapper .dataTables_paginate .paginate_button.current {
                background: #F72585 !important;
                color: #18122B !important;
                border: 1px solid #F72585 !important;
            }
            .dataTables_wrapper .dataTables_filter input {
                background: #18122B;
                color: #F7F7F7;
                border: 1px solid #7209B7;
                border-radius: 6px;
                padding: 2px 8px;
            }
            ::selection {
                background: #F72585;
                color: #18122B;
            }
            .view-toggle {
                margin: 10px 0;
            }
            .view-toggle a {
                color: #F72585;
                text-decoration: none;
                margin: 0 10px;
                font-weight: 600;
            }
            .view-toggle a.active {
                color: #B8B8FF;
                border-bottom: 2px solid #F72585;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{{coin_name}} <span style="color:#B8B8FF;">{{days}} Günlük Fiyat Tahmini</span></h1>
            <div class="view-toggle">
                <a href="?coin={{selected_coin}}&model={{selected_model}}&days={{days}}&view=single" class="{% if view=='single' %}active{% endif %}">Tek Model</a>
                <a href="?coin={{selected_coin}}&model={{selected_model}}&days={{days}}&view=compare" class="{% if view=='compare' %}active{% endif %}">Model Karşılaştırması</a>
            </div>
            <form method="get" class="select-coin">
                <label for="coin">Kripto Para Seçin:</label>
                <select name="coin" id="coin" onchange="this.form.submit()">
                    {% for cid, c in coins.items() %}
                        <option value="{{cid}}" {% if cid==selected_coin %}selected{% endif %}>{{c['name']}}</option>
                    {% endfor %}
                </select>
                <span class="model-select">
                <label for="model">Model:</label>
                <select name="model" id="model" onchange="this.form.submit()">
                    <option value="sarima" {% if selected_model=='sarima' %}selected{% endif %}>SARIMA</option>
                    {% if has_prophet %}
                    <option value="prophet" {% if selected_model=='prophet' %}selected{% endif %}>Prophet+Exog</option>
                    {% endif %}
                    {% if has_xgb %}
                    <option value="xgboost" {% if selected_model=='xgboost' %}selected{% endif %}>XGBoost</option>
                    {% endif %}
                    {% if has_torch %}
                    <option value="lstm" {% if selected_model=='lstm' %}selected{% endif %}>LSTM</option>
                    {% endif %}
                    <option value="ensemble" {% if selected_model=='ensemble' %}selected{% endif %}>Tüm Modeller (Ensemble)</option>
                </select>
                <label for="days">Gün:</label>
                <select name="days" id="days" onchange="this.form.submit()">
                    <option value="7" {% if days==7 %}selected{% endif %}>7</option>
                    <option value="30" {% if days==30 %}selected{% endif %}>30</option>
                    <option value="90" {% if days==90 %}selected{% endif %}>90</option>
                </select>
                <input type="hidden" name="view" value="{{view}}">
                </span>
            </form>
            <div id="main-graph">{{graph_html|safe}}</div>
            <div style="margin-bottom:16px;">
                <canvas id="sparkline" class="sparkline"></canvas>
            </div>
            <table id="priceTable" class="display" style="width:100%">
                <thead>
                    <tr>
                        <th>Tarih</th>
                        <th>Fiyat (USD)</th>
                        <th>Tip</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
            <div class="footer">
                <span style="color:#00F2FE;">Veriler: CoinGecko</span> |
                <span style="color:#F72585;">Model: <b>{{model_name}}</b></span> |
                <span style="color:#B8B8FF;"><b>{{today}}</b></span>
            </div>
        </div>
        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const tableData = {{table_html|safe}};
            $(document).ready(function() {
                $('#priceTable').DataTable({
                    data: tableData,
                    columns: [
                        { data: 'date' },
                        { data: 'price', render: $.fn.dataTable.render.number(',', '.', 2, '$') },
                        { data: 'type' }
                    ],
                    order: [[0, 'desc']],
                    pageLength: 10,
                    lengthChange: false,
                    searching: false,
                    info: false,
                    pagingType: "simple",
                    language: {
                        paginate: {
                            previous: "<span style='color:#F72585'>&lt;</span>",
                            next: "<span style='color:#F72585'>&gt;</span>"
                        }
                    }
                });
            });
            const sparkData = {{spark_data|tojson}};
            const ctx = document.getElementById('sparkline').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: sparkData.length}, (_, i) => i+1),
                    datasets: [{
                        data: sparkData,
                        borderColor: '#00F2FE',
                        backgroundColor: 'rgba(247,37,133,0.08)',
                        borderWidth: 2.5,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.45
                    }]
                },
                options: {
                    plugins: { legend: { display: false } },
                    scales: { x: { display: false }, y: { display: false } },
                    elements: { line: { borderJoinStyle: 'round' } }
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(
        html,
        coin_name=coin['name'],
        coins=COINS,
        selected_coin=coin_id,
        selected_model=model_type,
        has_prophet=has_prophet,
        has_xgb=has_xgb,
        has_lgb=has_lgb,
        has_torch=has_torch,
        model_name=model_name,
        graph_html=graph_html,
        today=datetime.now().strftime('%d.%m.%Y'),
        table_html=table_html,
        spark_data=spark_data,
        days=days,
        view=view
    )

if __name__ == '__main__':
    import os
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)