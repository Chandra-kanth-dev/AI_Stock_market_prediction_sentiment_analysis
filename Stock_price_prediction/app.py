# ============================================================
# AI POWERED STOCK MARKET PREDICTION PLATFORM
# Developed using Streamlit + NLP + ML + Finance
# Improvements: proper time-series CV, scaling, imbalance handling,
#               leakage-free features, SHAP fix, robust yfinance parsing
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import shap

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from dotenv import load_dotenv

from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ============================================================
# APP HEADER
# ============================================================

st.title("📈 AI Powered Stock Market Prediction Platform")
st.markdown("""
End-to-End Data Science Project | Features: Financial News · NLP Sentiment · Technical Indicators · ML · Backtesting · Prediction Dashboard
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")

start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2021-01-01")
)

end_date = st.sidebar.date_input(
    "End Date", value=pd.to_datetime("today")
)

prediction_horizon = st.sidebar.slider(
    "Prediction Horizon (days)", min_value=1, max_value=10, value=5,
    help="Predict if price will be higher after N days"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
use_smote = st.sidebar.checkbox(
    "Handle Class Imbalance (SMOTE)", value=True,
    help="Use SMOTE to balance classes. Disable for large datasets."
)
n_cv_splits = st.sidebar.slider("Cross-Validation Splits", 3, 7, 5)

# ============================================================
# DATA COLLECTION
# ============================================================

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df.empty:
            st.error(f"No data found for {ticker}. Check the ticker symbol.")
            st.stop()

        df.reset_index(inplace=True)

        # Robustly flatten MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0]
                          for col in df.columns]

        # Ensure standard column names
        df.columns = [str(c).strip() for c in df.columns]

        # Rename 'Adj Close' if present
        if 'Adj Close' in df.columns and 'Close' not in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)

        return df

    except Exception as e:
        st.error(f"Download Error: {e}")
        st.stop()


def clean_stock_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    df.columns = [str(col).replace(" ", "_") for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


df = load_stock_data(ticker, start_date, end_date)
df = clean_stock_data(df)

# ============================================================
# BASIC OVERVIEW
# ============================================================

st.subheader("Stock Dataset")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric("Columns", len(df.columns))
with col3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

fig = px.line(df, x="Date", y="Close", title=f"{ticker} Closing Price")
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# NEWS API INTEGRATION (optional)
# ============================================================

@st.cache_data
def fetch_news(ticker):
    if not NEWS_API_KEY:
        return pd.DataFrame()
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        news = newsapi.get_everything(
            q=ticker, language="en",
            sort_by="publishedAt", page_size=100
        )
        return pd.DataFrame(news.get("articles", []))
    except Exception as e:
        st.warning(f"News API Error: {e}")
        return pd.DataFrame()


def clean_news(news_df):
    if news_df.empty:
        return news_df
    news_df = news_df.copy()
    cols = [c for c in ["publishedAt", "title", "description", "source"]
            if c in news_df.columns]
    news_df = news_df[cols]
    if "source" in news_df.columns:
        news_df["source"] = news_df["source"].apply(
            lambda x: x["name"] if isinstance(x, dict) else str(x)
        )
    news_df.drop_duplicates(subset=["title"], inplace=True)
    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"])
    news_df["Date"] = news_df["publishedAt"].dt.date
    return news_df


news_df = fetch_news(ticker)
news_df = clean_news(news_df)

st.subheader("Latest Financial News")
if not news_df.empty:
    st.dataframe(news_df.head(20))
else:
    st.info("No news available. Add NEWS_API_KEY to .env to enable sentiment features.")

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    return analyzer.polarity_scores(str(text))["compound"]


def sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


if not news_df.empty:
    news_df["sentiment"] = news_df["title"].astype(str).apply(get_sentiment_score)
    news_df["label"] = news_df["sentiment"].apply(sentiment_label)

    st.subheader("Sentiment Overview")
    counts = news_df["label"].value_counts()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Positive", counts.get("Positive", 0))
    with c2:
        st.metric("Negative", counts.get("Negative", 0))
    with c3:
        st.metric("Neutral", counts.get("Neutral", 0))

    sentiment_fig = px.pie(
        names=counts.index, values=counts.values,
        title="Sentiment Distribution"
    )
    st.plotly_chart(sentiment_fig, use_container_width=True)

    daily_sentiment = (
        news_df.groupby("Date")["sentiment"].mean().reset_index()
    )
    daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"])
else:
    daily_sentiment = pd.DataFrame(columns=["Date", "sentiment"])

# ============================================================
# MERGE SENTIMENT — USE ONLY LAGGED VALUES TO PREVENT LEAKAGE
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df = pd.merge(df, daily_sentiment, on="Date", how="left")
df["sentiment"] = df["sentiment"].fillna(0)

# CRITICAL: All sentiment features must be lagged to prevent data leakage
# We never use same-day sentiment as a feature — only past sentiment
df["sentiment_lag1"] = df["sentiment"].shift(1)
df["sentiment_lag3"] = df["sentiment"].shift(3)
df["sentiment_rolling5"] = df["sentiment"].shift(1).rolling(5).mean()
df["sentiment_rolling10"] = df["sentiment"].shift(1).rolling(10).mean()
df["sentiment_change"] = df["sentiment"].shift(1).diff()

# Drop current-day raw sentiment from features (only lags will be used)
df.drop(columns=["sentiment"], inplace=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

st.subheader("Feature Engineering")

# Moving Averages
for window in [5, 7, 14, 21, 50]:
    df[f"MA_{window}"] = df["Close"].rolling(window).mean()

# EMA
for span in [5, 12, 26, 50]:
    df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

# MACD
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

# RSI
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()   # Wilder's smoothing (more accurate)
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
df["RSI"] = 100 - (100 / (1 + rs))

# Stochastic RSI
rsi_min = df["RSI"].rolling(14).min()
rsi_max = df["RSI"].rolling(14).max()
df["STOCH_RSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

# Bollinger Bands
bb_mean = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()
df["BB_UPPER"] = bb_mean + 2 * bb_std
df["BB_LOWER"] = bb_mean - 2 * bb_std
df["BB_WIDTH"] = df["BB_UPPER"] - df["BB_LOWER"]
df["BB_PCT"] = (df["Close"] - df["BB_LOWER"]) / (df["BB_WIDTH"] + 1e-9)  # Where price is within bands

# Returns
df["Daily_Return"] = df["Close"].pct_change()
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df["Price_Change_Pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

# Volatility
for window in [5, 10, 21]:
    df[f"Volatility_{window}"] = df["Daily_Return"].rolling(window).std()

# Momentum
for lag in [3, 7, 14, 21]:
    df[f"Momentum_{lag}"] = df["Close"] - df["Close"].shift(lag)

# Rate of Change
for lag in [5, 10]:
    df[f"ROC_{lag}"] = (df["Close"] - df["Close"].shift(lag)) / df["Close"].shift(lag) * 100

# Volume Features
df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
df["Volume_MA_21"] = df["Volume"].rolling(21).mean()
df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA_21"] + 1e-9)   # Normalized volume
df["Volume_Change"] = df["Volume"].pct_change()

# Lag Features (Close price lags — these are safe)
for lag in [1, 2, 3, 5, 7, 14]:
    df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

# High/Low/Spread
df["High_Low_Spread"] = df["High"] - df["Low"]
df["Open_Close_Spread"] = df["Close"] - df["Open"]

# Rolling Stats
df["Rolling_Max_21"] = df["High"].rolling(21).max()
df["Rolling_Min_21"] = df["Low"].rolling(21).min()
df["Rolling_Mean_21"] = df["Close"].rolling(21).mean()
df["Distance_From_High"] = (df["Close"] - df["Rolling_Max_21"]) / df["Rolling_Max_21"]
df["Distance_From_Low"] = (df["Close"] - df["Rolling_Min_21"]) / df["Rolling_Min_21"]

# ATR (Average True Range)
high_low = df["High"] - df["Low"]
high_close = np.abs(df["High"] - df["Close"].shift())
low_close = np.abs(df["Low"] - df["Close"].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["ATR"] = true_range.rolling(14).mean()
df["ATR_Normalized"] = df["ATR"] / df["Close"]  # Normalized ATR

# Composite: Fear-Greed proxy (only uses lagged sentiment)
df["Fear_Greed"] = (df["RSI"] / 100) + df["sentiment_lag1"].fillna(0)

# ============================================================
# TARGET VARIABLE — configurable horizon, no leakage
# ============================================================

df["Target"] = (
    df["Close"].shift(-prediction_horizon) > df["Close"]
).astype(int)

# ============================================================
# DROP NULLS AFTER FEATURE ENGINEERING
# ============================================================

df = df.dropna().reset_index(drop=True)

numeric_features = df.select_dtypes(include=np.number).columns
st.success(f"Total Engineered Features: {len(numeric_features)}")
st.dataframe(df.head())

# Check class balance
class_counts = df["Target"].value_counts()
st.info(f"Class Distribution — UP: {class_counts.get(1,0)} | DOWN: {class_counts.get(0,0)} "
        f"| Balance Ratio: {class_counts.min()/class_counts.max():.2f}")

# ============================================================
# EDA DASHBOARD
# ============================================================

st.header("📊 Exploratory Data Analysis")

# Candlestick
st.subheader("Candlestick Chart")
candlestick_fig = go.Figure(data=[go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"]
)])
candlestick_fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(candlestick_fig, use_container_width=True)

# Moving Averages
st.subheader("Moving Averages")
ma_fig = go.Figure()
for col, name in [("Close", "Close"), ("MA_21", "MA 21"), ("MA_50", "MA 50")]:
    if col in df.columns:
        ma_fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=name))
st.plotly_chart(ma_fig, use_container_width=True)

# Volume
st.subheader("Volume Analysis")
volume_fig = px.bar(df, x="Date", y="Volume", title="Daily Trading Volume")
st.plotly_chart(volume_fig, use_container_width=True)

# RSI
st.subheader("RSI Indicator")
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
rsi_fig.add_hline(y=70, line_color="red", annotation_text="Overbought")
rsi_fig.add_hline(y=30, line_color="green", annotation_text="Oversold")
st.plotly_chart(rsi_fig, use_container_width=True)

# MACD
st.subheader("MACD Indicator")
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], name="Signal"))
macd_fig.add_bar(x=df["Date"], y=df["MACD_HIST"], name="Histogram")
st.plotly_chart(macd_fig, use_container_width=True)

# Bollinger Bands
st.subheader("Bollinger Bands")
bb_fig = go.Figure()
bb_fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
bb_fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], name="Upper Band",
                             line=dict(dash="dash")))
bb_fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], name="Lower Band",
                             line=dict(dash="dash"), fill='tonexty', fillcolor='rgba(0,100,255,0.05)'))
st.plotly_chart(bb_fig, use_container_width=True)

# Returns Distribution
st.subheader("Daily Return Distribution")
return_fig = px.histogram(df, x="Daily_Return", nbins=60, title="Daily Return Distribution")
st.plotly_chart(return_fig, use_container_width=True)

# Volatility
st.subheader("Volatility Analysis")
vol_fig = px.line(df, x="Date", y="Volatility_21", title="21-Day Rolling Volatility")
st.plotly_chart(vol_fig, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=np.number)
# Only show top correlated features to avoid giant unreadable chart
top_corr_features = (
    numeric_df.corr()["Target"].abs()
    .sort_values(ascending=False)
    .head(25).index.tolist()
)
corr_matrix = numeric_df[top_corr_features].corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax, annot=False)
ax.set_title("Top 25 Features Correlation Heatmap")
st.pyplot(fig)

# Top Correlations with Target
st.subheader("Most Important Correlations with Target")
target_corr = (
    numeric_df.corr()["Target"]
    .sort_values(ascending=False)
    .reset_index()
)
target_corr.columns = ["Feature", "Correlation"]
st.dataframe(target_corr.head(20))

# Feature Distribution Explorer
st.subheader("Feature Distribution Explorer")
feature_choice = st.selectbox("Choose Feature", numeric_df.columns)
dist_fig = px.histogram(df, x=feature_choice, nbins=50, title=f"{feature_choice} Distribution")
st.plotly_chart(dist_fig, use_container_width=True)

# Summary Statistics
st.subheader("Summary Statistics")
st.dataframe(df.describe())

# ============================================================
# MACHINE LEARNING
# ============================================================

st.header("🤖 Machine Learning Models")

# ============================================================
# FEATURE SELECTION — exclude raw OHLCV and non-numeric
# Raw price columns (Open, High, Low, Close, Volume) are excluded
# to prevent the model from trivially fitting on price levels.
# Only derived/normalized features are used.
# ============================================================

EXCLUDE_COLS = [
    "Date", "Target",
    "Open", "High", "Low", "Close", "Volume",  # raw prices — use derived only
]

features = [col for col in df.columns if col not in EXCLUDE_COLS
            and df[col].dtype in [np.float64, np.int64, float, int]]

X = df[features]
y = df["Target"]

# ============================================================
# TIME-SERIES SPLIT — never shuffle, preserve temporal order
# ============================================================

split_index = int(len(df) * 0.80)
X_train_raw = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

st.success(f"Training Rows: {len(X_train_raw)} | Test Rows: {len(X_test)}")

# ============================================================
# HANDLE CLASS IMBALANCE WITH SMOTE (train set only!)
# ============================================================

if use_smote:
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train_bal = smote.fit_resample(X_train_raw, y_train)
        st.info(f"SMOTE applied — Training set after balancing: {len(X_train)} rows")
    except Exception as e:
        st.warning(f"SMOTE failed ({e}), using original training data.")
        X_train, y_train_bal = X_train_raw, y_train
else:
    X_train, y_train_bal = X_train_raw, y_train

# ============================================================
# FEATURE SCALING (critical for Logistic Regression)
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TIME-SERIES CROSS-VALIDATION FUNCTION
# ============================================================

def ts_cross_validate(model, X, y, n_splits=5):
    """Proper time-series cross-validation — no data leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc", n_jobs=-1)
    return scores


# ============================================================
# LOGISTIC REGRESSION (with scaling)
# ============================================================

with st.spinner("Training Logistic Regression..."):
    lr_model = LogisticRegression(
        max_iter=5000, C=0.1, solver="lbfgs",
        class_weight="balanced", random_state=42
    )
    lr_model.fit(X_train_scaled, y_train_bal)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_cv_scores = ts_cross_validate(
        Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(
            max_iter=5000, C=0.1, class_weight="balanced", random_state=42
        ))]),
        X_train_raw, y_train, n_splits=n_cv_splits
    )

# ============================================================
# RANDOM FOREST
# ============================================================

with st.spinner("Training Random Forest..."):
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=10,         # Prevent overfitting
        max_features="sqrt",
        class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train_bal)   # Tree models don't need scaling
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_cv_scores = ts_cross_validate(
        RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        X_train_raw, y_train, n_splits=n_cv_splits
    )

# ============================================================
# XGBOOST with early stopping via eval set
# ============================================================

with st.spinner("Training XGBoost..."):
    # Use last 10% of train for early stopping validation
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train_bal[:val_split], y_train_bal[val_split:]

    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    xgb_model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,       # Regularization
        gamma=0.1,                 # Min loss reduction for split
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=1.0,            # L2 regularization
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=30,
        verbosity=0
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_cv_scores = ts_cross_validate(
        XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            random_state=42, eval_metric="logloss", verbosity=0
        ),
        X_train_raw, y_train, n_splits=n_cv_splits
    )

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_name, y_true, predictions, probabilities, cv_scores):
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, predictions), 4),
        "Precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "Recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "F1": round(f1_score(y_true, predictions, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, probabilities), 4),
        "CV_AUC_Mean": round(cv_scores.mean(), 4),
        "CV_AUC_Std": round(cv_scores.std(), 4),
    }


leaderboard = pd.DataFrame([
    evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob, lr_cv_scores),
    evaluate_model("Random Forest", y_test, rf_pred, rf_prob, rf_cv_scores),
    evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob, xgb_cv_scores),
]).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

st.subheader("Model Leaderboard (with Time-Series CV)")
st.dataframe(leaderboard)
st.caption("CV_AUC_Mean = mean ROC-AUC across time-series folds (more reliable than single split)")

# ============================================================
# BEST MODEL SELECTION
# ============================================================

best_model_name = leaderboard.iloc[0]["Model"]
st.success(f"Best Model: {best_model_name}")

if best_model_name == "XGBoost":
    best_model = xgb_model
    best_pred = xgb_pred
    best_prob = xgb_prob
    X_test_for_model = X_test        # Tree models use unscaled
    X_train_for_model = X_train
elif best_model_name == "Random Forest":
    best_model = rf_model
    best_pred = rf_pred
    best_prob = rf_prob
    X_test_for_model = X_test
    X_train_for_model = X_train
else:
    best_model = lr_model
    best_pred = lr_pred
    best_prob = lr_prob
    X_test_for_model = X_test_scaled  # LR uses scaled
    X_train_for_model = X_train_scaled

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(y_test, best_pred)
prec = precision_score(y_test, best_pred, zero_division=0)
rec = recall_score(y_test, best_pred, zero_division=0)
f1 = f1_score(y_test, best_pred, zero_division=0)
auc = roc_auc_score(y_test, best_prob)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Accuracy", f"{acc:.3f}")
with c2: st.metric("Precision", f"{prec:.3f}")
with c3: st.metric("Recall", f"{rec:.3f}")
with c4: st.metric("F1 Score", f"{f1:.3f}")
with c5: st.metric("ROC AUC", f"{auc:.3f}")

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, best_pred)
cm_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                   labels={"x": "Predicted", "y": "Actual"},
                   x=["DOWN", "UP"], y=["DOWN", "UP"])
st.plotly_chart(cm_fig, use_container_width=True)

# ROC Curve — compare all models
st.subheader("ROC Curve Comparison")
roc_fig = go.Figure()
for name, y_prob in [
    ("Logistic Regression", lr_prob),
    ("Random Forest", rf_prob),
    ("XGBoost", xgb_prob)
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"{name} (AUC={auc_val:.3f})"))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash"), name="Random Baseline"))
roc_fig.update_layout(xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate", height=500)
st.plotly_chart(roc_fig, use_container_width=True)

# Cross-Validation Scores Visualization
st.subheader("Cross-Validation AUC by Fold")
cv_data = []
for name, scores in [
    ("Logistic Regression", lr_cv_scores),
    ("Random Forest", rf_cv_scores),
    ("XGBoost", xgb_cv_scores)
]:
    for i, s in enumerate(scores):
        cv_data.append({"Model": name, "Fold": f"Fold {i+1}", "AUC": s})

cv_df = pd.DataFrame(cv_data)
cv_fig = px.bar(cv_df, x="Fold", y="AUC", color="Model", barmode="group",
                title="Time-Series CV AUC per Fold")
st.plotly_chart(cv_fig, use_container_width=True)

# Feature Importance
if best_model_name in ["XGBoost", "Random Forest"]:
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    st.dataframe(importance_df.head(25))
    importance_fig = px.bar(
        importance_df.head(20), x="Importance", y="Feature",
        orientation="h", title="Top 20 Features"
    )
    st.plotly_chart(importance_fig, use_container_width=True)

# ============================================================
# PREDICTION ENGINE
# ============================================================

st.header("🔮 Tomorrow Market Prediction")

if best_model_name in ["XGBoost", "Random Forest"]:
    latest_data = X.iloc[-1:]
else:
    latest_data = scaler.transform(X.iloc[-1:])

if best_model_name in ["XGBoost", "Random Forest"]:
    prediction = best_model.predict(X.iloc[-1:])[0]
    prediction_probabilities = best_model.predict_proba(X.iloc[-1:])[0]
else:
    prediction = best_model.predict(scaler.transform(X.iloc[-1:]))[0]
    prediction_probabilities = best_model.predict_proba(scaler.transform(X.iloc[-1:]))[0]

down_probability = prediction_probabilities[0]
up_probability = prediction_probabilities[1]
confidence = max(prediction_probabilities) * 100

prediction_col1, prediction_col2 = st.columns(2)
with prediction_col1:
    if prediction == 1:
        st.success(f"📈 PREDICTION: STOCK LIKELY TO MOVE UP (next {prediction_horizon} days)")
    else:
        st.error(f"📉 PREDICTION: STOCK LIKELY TO MOVE DOWN (next {prediction_horizon} days)")
with prediction_col2:
    st.metric("Confidence Score", f"{confidence:.2f}%")

c1, c2 = st.columns(2)
with c1: st.metric("UP Probability", f"{up_probability*100:.2f}%")
with c2: st.metric("DOWN Probability", f"{down_probability*100:.2f}%")

# Gauge
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=up_probability * 100,
    title={"text": "Bullish Probability (%)"},
    delta={"reference": 50},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"thickness": 0.3},
        "steps": [
            {"range": [0, 40], "color": "rgba(255,0,0,0.2)"},
            {"range": [40, 60], "color": "rgba(255,255,0,0.2)"},
            {"range": [60, 100], "color": "rgba(0,255,0,0.2)"},
        ],
        "threshold": {"line": {"color": "black", "width": 4}, "value": 50}
    }
))
st.plotly_chart(gauge_fig, use_container_width=True)

# Trading Signal
st.subheader("Trading Signal")
latest_rsi = df["RSI"].iloc[-1]
latest_macd = df["MACD"].iloc[-1]
latest_signal_val = df["MACD_SIGNAL"].iloc[-1]
latest_sentiment = df["sentiment_lag1"].iloc[-1]

signal = "HOLD"
if latest_rsi < 30 and latest_macd > latest_signal_val:
    signal = "BUY"
elif latest_rsi > 70 and latest_macd < latest_signal_val:
    signal = "SELL"

if signal == "BUY":
    st.success("🟢 BUY SIGNAL")
elif signal == "SELL":
    st.error("🔴 SELL SIGNAL")
else:
    st.warning("🟡 HOLD SIGNAL")

# Market Snapshot
st.subheader("Market Snapshot")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Latest Close", round(df["Close"].iloc[-1], 2))
with col2: st.metric("RSI", round(latest_rsi, 2))
with col3: st.metric("MACD", round(latest_macd, 4))
with col4: st.metric("Sentiment (lag1)", round(latest_sentiment, 3))

# Market Health Score
health_score = sum([
    25 if latest_sentiment > 0 else 0,
    25 if latest_rsi < 70 else 0,
    25 if latest_macd > latest_signal_val else 0,
    25 if prediction == 1 else 0,
])
st.subheader("Market Health Score")
st.progress(health_score / 100)
st.metric("Score", f"{health_score}/100")

if health_score >= 75:
    st.success("Strong Bullish Setup — Positive sentiment · Healthy indicators · Model predicts up · Moderate risk")
elif health_score >= 50:
    st.info("Neutral Market — Mixed signals · Wait for confirmation · Monitor momentum")
else:
    st.warning("Bearish Conditions — Weak technical structure · Negative sentiment · Elevated downside risk")

# Latest Feature Values
st.subheader("Latest Feature Values Used for Prediction")
latest_feature_df = pd.DataFrame({
    "Feature": X.columns, "Value": X.iloc[-1].values
})
st.dataframe(latest_feature_df)

# Recent Prices
st.subheader("Recent Close Prices")
history_fig = px.line(df.tail(100), x="Date", y="Close", title="Last 100 Trading Days")
st.plotly_chart(history_fig, use_container_width=True)

# ============================================================
# BACKTESTING ENGINE
# ============================================================

st.header("📊 Strategy Backtesting")

test_predictions = best_pred
backtest_df = df.iloc[split_index:].copy().reset_index(drop=True)
backtest_df["Prediction"] = test_predictions

backtest_df["Market_Return"] = backtest_df["Close"].pct_change()
backtest_df["Strategy_Return"] = backtest_df["Prediction"].shift(1) * backtest_df["Market_Return"]
backtest_df.dropna(inplace=True)

backtest_df["Cumulative_Market"] = (1 + backtest_df["Market_Return"]).cumprod()
backtest_df["Cumulative_Strategy"] = (1 + backtest_df["Strategy_Return"]).cumprod()

st.subheader("Strategy vs Market")
performance_fig = go.Figure()
performance_fig.add_trace(go.Scatter(
    x=backtest_df["Date"], y=backtest_df["Cumulative_Market"], name="Buy & Hold Market"
))
performance_fig.add_trace(go.Scatter(
    x=backtest_df["Date"], y=backtest_df["Cumulative_Strategy"], name="ML Strategy"
))
performance_fig.update_layout(height=600)
st.plotly_chart(performance_fig, use_container_width=True)

market_return = (backtest_df["Cumulative_Market"].iloc[-1] - 1) * 100
strategy_return = (backtest_df["Cumulative_Strategy"].iloc[-1] - 1) * 100

strategy_std = backtest_df["Strategy_Return"].std()
sharpe_ratio = (
    (backtest_df["Strategy_Return"].mean() / strategy_std) * np.sqrt(252)
    if strategy_std != 0 else 0
)

rolling_max = backtest_df["Cumulative_Strategy"].cummax()
drawdown = (backtest_df["Cumulative_Strategy"] / rolling_max) - 1
max_drawdown = drawdown.min() * 100

win_rate = (backtest_df["Strategy_Return"] > 0).sum() / len(backtest_df) * 100

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Market Return", f"{market_return:.2f}%")
with k2: st.metric("Strategy Return", f"{strategy_return:.2f}%")
with k3: st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
with k4: st.metric("Win Rate", f"{win_rate:.2f}%")
st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

st.subheader("Drawdown Analysis")
drawdown_fig = px.area(x=backtest_df["Date"], y=drawdown, title="Portfolio Drawdown")
st.plotly_chart(drawdown_fig, use_container_width=True)

backtest_df["Month"] = pd.to_datetime(backtest_df["Date"]).dt.to_period("M").astype(str)
monthly_returns = backtest_df.groupby("Month")["Strategy_Return"].sum().reset_index()
monthly_fig = px.bar(monthly_returns, x="Month", y="Strategy_Return",
                     title="Monthly Strategy Returns",
                     color="Strategy_Return",
                     color_continuous_scale=["red", "green"])
st.plotly_chart(monthly_fig, use_container_width=True)

# Download backtest
st.download_button(
    "Download Backtest CSV",
    backtest_df.to_csv(index=False),
    file_name=f"{ticker}_backtest.csv",
    mime="text/csv"
)

# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================

st.header("💼 Portfolio Analytics")

comparison_tickers = st.multiselect(
    "Compare Stocks",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    default=[ticker]
)

if comparison_tickers:
    comparison_data = {}
    for stock in comparison_tickers:
        try:
            temp_df = yf.download(
                stock, start=start_date, end=end_date,
                auto_adjust=True, progress=False
            )
            if not temp_df.empty:
                # Robust MultiIndex flattening
                if isinstance(temp_df.columns, pd.MultiIndex):
                    temp_df.columns = [col[0] for col in temp_df.columns]
                comparison_data[stock] = temp_df["Close"].values[:len(temp_df)]
        except Exception:
            pass

    if comparison_data:
        max_len = min(len(v) for v in comparison_data.values())
        comp_df = pd.DataFrame(
            {k: v[:max_len] for k, v in comparison_data.items()}
        )
        normalized_df = comp_df / comp_df.iloc[0] * 100
        comparison_fig = px.line(normalized_df, title="Normalized Performance (Base=100)")
        st.plotly_chart(comparison_fig, use_container_width=True)

# Portfolio Simulator
st.subheader("Portfolio Simulator")
investment_amount = st.number_input("Investment Amount (₹/$)", min_value=100, value=10000)
future_value = investment_amount * (1 + strategy_return / 100)
alpha = strategy_return - market_return

c1, c2, c3 = st.columns(3)
with c1: st.metric("Initial Investment", f"{investment_amount:,.0f}")
with c2: st.metric("Projected Value", f"{future_value:,.0f}")
with c3: st.metric("Alpha vs Market", f"{alpha:.2f}%")

# ============================================================
# RISK DASHBOARD
# ============================================================

st.header("⚠️ Risk Dashboard")

risk_score = sum([
    25 if latest_rsi > 70 else 0,
    25 if latest_sentiment < 0 else 0,
    25 if max_drawdown < -20 else 0,
    25 if sharpe_ratio < 1 else 0,
])

risk_level = (
    "Low" if risk_score <= 25 else
    "Moderate" if risk_score <= 50 else
    "High" if risk_score <= 75 else "Very High"
)

st.metric("Risk Level", risk_level)
st.progress(risk_score / 100)

risk_df = pd.DataFrame({
    "Metric": ["RSI", "Sentiment (lag1)", "Max Drawdown %", "Sharpe Ratio"],
    "Value": [round(latest_rsi, 2), round(latest_sentiment, 3),
              round(max_drawdown, 2), round(sharpe_ratio, 2)],
    "Signal": [
        "⚠️ Overbought" if latest_rsi > 70 else "✅ Normal",
        "⚠️ Negative" if latest_sentiment < 0 else "✅ Positive",
        "⚠️ High Drawdown" if max_drawdown < -20 else "✅ Acceptable",
        "⚠️ Low" if sharpe_ratio < 1 else "✅ Good",
    ]
})
st.dataframe(risk_df)

# ============================================================
# EXPLAINABLE AI — SHAP
# ============================================================

st.header("🧠 Explainable AI (SHAP)")

if best_model_name in ["XGBoost", "Random Forest"]:
    try:
        st.subheader("SHAP Feature Importance")
        sample_data = X_test.tail(min(200, len(X_test)))
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(sample_data)

        # Handle binary classification: RF returns list, XGB returns array
        if isinstance(shap_values, list):
            # For RF binary: shap_values[1] = positive class
            sv = shap_values[1]
        else:
            sv = shap_values

        shap_importance = pd.DataFrame({
            "Feature": sample_data.columns,
            "SHAP_Importance": np.abs(sv).mean(axis=0)
        }).sort_values("SHAP_Importance", ascending=False)

        st.dataframe(shap_importance.head(20))

        shap_fig = px.bar(
            shap_importance.head(20), x="SHAP_Importance", y="Feature",
            orientation="h", title="SHAP Global Feature Importance (Mean |SHAP|)"
        )
        st.plotly_chart(shap_fig, use_container_width=True)

    except Exception as e:
        st.warning(f"SHAP Error: {e}")

# ============================================================
# PREDICTION CONFIDENCE DISTRIBUTION
# ============================================================

st.header("🎯 Prediction Confidence Analysis")

confidence_series = np.max(best_model.predict_proba(X_test), axis=1) * 100
confidence_df = pd.DataFrame({"Confidence": confidence_series})
confidence_fig = px.histogram(
    confidence_df, x="Confidence", nbins=30,
    title="Prediction Confidence Distribution"
)
st.plotly_chart(confidence_fig, use_container_width=True)

avg_conf = confidence_series.mean()
high_conf_pct = (confidence_series >= 70).mean() * 100
st.info(f"Average Confidence: {avg_conf:.1f}% | High-Confidence Predictions (≥70%): {high_conf_pct:.1f}%")

# ============================================================
# MODEL ERRORS
# ============================================================

st.header("❌ Model Errors Analysis")

error_df = X_test.copy()
error_df["Actual"] = y_test.values
error_df["Predicted"] = best_pred
error_df["Confidence"] = np.max(best_model.predict_proba(X_test), axis=1)

errors = error_df[error_df["Actual"] != error_df["Predicted"]]
st.write(f"Misclassified Samples: {len(errors)} / {len(error_df)} ({len(errors)/len(error_df)*100:.1f}%)")
st.dataframe(errors[["Actual", "Predicted", "Confidence"] + list(X.columns[:5])].head(50))

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.header("📋 Executive Summary")

summary_text = f"""
Ticker:             {ticker}
Best Model:         {best_model_name}
Prediction:         {'UP' if prediction == 1 else 'DOWN'} (next {prediction_horizon} days)
Confidence:         {confidence:.2f}%

Model Performance:
  Accuracy:         {acc:.4f}
  ROC AUC:          {auc:.4f}
  F1 Score:         {f1:.4f}

Strategy Performance:
  Strategy Return:  {strategy_return:.2f}%
  Market Return:    {market_return:.2f}%
  Alpha:            {strategy_return - market_return:.2f}%
  Sharpe Ratio:     {sharpe_ratio:.2f}
  Win Rate:         {win_rate:.2f}%
  Max Drawdown:     {max_drawdown:.2f}%

Market Conditions:
  RSI:              {latest_rsi:.2f}
  Signal:           {signal}
  Risk Level:       {risk_level}
  Health Score:     {health_score}/100
"""

st.text_area("Summary", summary_text, height=320)

# ============================================================
# EXPORT
# ============================================================

st.header("📥 Export")
st.download_button(
    "Download Full Dataset", df.to_csv(index=False),
    file_name=f"{ticker}_full_dataset.csv", mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
### AI Powered Stock Market Prediction Platform

✅ Data Collection · ✅ Sentiment Analysis · ✅ Feature Engineering (leakage-free)
✅ Technical Indicators · ✅ ML with Time-Series CV · ✅ Class Imbalance Handling
✅ SHAP Explainability · ✅ Backtesting · ✅ Portfolio Analytics · ✅ Risk Dashboard

> ⚠️ **Disclaimer**: This platform is for educational purposes only. Stock predictions are inherently uncertain. Do not make real investment decisions based on model output alone.
""")
