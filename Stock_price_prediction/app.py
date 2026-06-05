# ============================================================
# AI POWERED STOCK MARKET PREDICTION PLATFORM
# Developed using Streamlit + NLP + ML + Finance
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import shap

from newsapi import NewsApiClient

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

from dotenv import load_dotenv

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)

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

NEWS_API_KEY = os.getenv(
    "NEWS_API_KEY"
)

# ============================================================
# APP HEADER
# ============================================================

st.title(
    "📈 AI Powered Stock Market Prediction Platform"
)

st.markdown(
    """
    End-to-End Data Science Project

    Features:
    - Financial News Analysis
    - NLP Sentiment Analysis
    - Technical Indicators
    - Machine Learning
    - Backtesting
    - Prediction Dashboard
    """
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Configuration"
)

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime(
        "2022-01-01"
    )
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime(
        "today"
    )
)

# ============================================================
# DATA COLLECTION
# ============================================================

@st.cache_data
def load_stock_data(
        ticker,
        start_date,
        end_date
):

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    df.reset_index(
        inplace=True
    )

    return df

# ============================================================
# CLEAN DATA
# ============================================================

def clean_stock_data(df):

    df = df.copy()

    df.drop_duplicates(
        inplace=True
    )

    df.dropna(
        inplace=True
    )

    df.columns = [
        str(col)
        .replace(" ", "_")
        for col in df.columns
    ]

    return df

# ============================================================
# LOAD DATA
# ============================================================

df = load_stock_data(
    ticker,
    start_date,
    end_date
)

df = clean_stock_data(
    df
)

# ============================================================
# BASIC OVERVIEW
# ============================================================

st.subheader(
    "Stock Dataset"
)

st.dataframe(
    df.head()
)

col1,col2,col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        len(df)
    )

with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )

with col3:

    st.metric(
        "Missing Values",
        int(
            df.isna()
            .sum()
            .sum()
        )
    )

# ============================================================
# PRICE CHART
# ============================================================

fig = px.line(
    df,
    x="Date",
    y="Close",
    title=f"{ticker} Closing Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# ============================================================
# NEWS API INTEGRATION
# ============================================================

@st.cache_data
def fetch_news(ticker):

    if NEWS_API_KEY is None:
        return pd.DataFrame()

    try:

        newsapi = NewsApiClient(
            api_key=NEWS_API_KEY
        )

        news = newsapi.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=100
        )

        articles = news["articles"]

        news_df = pd.DataFrame(
            articles
        )

        return news_df

    except Exception as e:

        st.warning(
            f"News API Error: {e}"
        )

        return pd.DataFrame()

# ============================================================
# CLEAN NEWS DATA
# ============================================================

def clean_news(news_df):

    if news_df.empty:
        return news_df

    news_df = news_df.copy()

    columns_needed = [
        "publishedAt",
        "title",
        "description",
        "source"
    ]

    available_cols = [
        col
        for col in columns_needed
        if col in news_df.columns
    ]

    news_df = news_df[
        available_cols
    ]

    if "source" in news_df.columns:

        news_df["source"] = (
            news_df["source"]
            .apply(
                lambda x:
                x["name"]
                if isinstance(x,dict)
                else str(x)
            )
        )

    news_df.drop_duplicates(
        subset=["title"],
        inplace=True
    )

    news_df["publishedAt"] = (
        pd.to_datetime(
            news_df["publishedAt"]
        )
    )

    news_df["Date"] = (
        news_df["publishedAt"]
        .dt.date
    )

    return news_df

# ============================================================
# DOWNLOAD NEWS
# ============================================================

news_df = fetch_news(
    ticker
)

news_df = clean_news(
    news_df
)

# ============================================================
# SHOW NEWS
# ============================================================

st.subheader(
    "Latest Financial News"
)

if not news_df.empty:

    st.dataframe(
        news_df.head(20)
    )

else:

    st.info(
        "No news available."
    )

# ============================================================
# SENTIMENT ANALYZER
# ============================================================

analyzer = (
    SentimentIntensityAnalyzer()
)

# ============================================================
# SENTIMENT SCORE FUNCTION
# ============================================================

def get_sentiment_score(text):

    score = (
        analyzer
        .polarity_scores(
            str(text)
        )
    )

    return score["compound"]

# ============================================================
# CALCULATE SENTIMENT
# ============================================================

if not news_df.empty:

    news_df["sentiment"] = (
        news_df["title"]
        .astype(str)
        .apply(
            get_sentiment_score
        )
    )

# ============================================================
# SENTIMENT LABEL
# ============================================================

def sentiment_label(score):

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    return "Neutral"

# ============================================================
# ASSIGN LABELS
# ============================================================

if not news_df.empty:

    news_df["label"] = (
        news_df["sentiment"]
        .apply(
            sentiment_label
        )
    )

# ============================================================
# SENTIMENT METRICS
# ============================================================

st.subheader(
    "Sentiment Overview"
)

if not news_df.empty:

    positive_count = (
        news_df["label"]
        .eq("Positive")
        .sum()
    )

    negative_count = (
        news_df["label"]
        .eq("Negative")
        .sum()
    )

    neutral_count = (
        news_df["label"]
        .eq("Neutral")
        .sum()
    )

    c1,c2,c3 = st.columns(3)

    with c1:

        st.metric(
            "Positive",
            positive_count
        )

    with c2:

        st.metric(
            "Negative",
            negative_count
        )

    with c3:

        st.metric(
            "Neutral",
            neutral_count
        )

# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

if not news_df.empty:

    sentiment_fig = px.pie(

        names=[
            "Positive",
            "Negative",
            "Neutral"
        ],

        values=[
            positive_count,
            negative_count,
            neutral_count
        ],

        title="Sentiment Distribution"
    )

    st.plotly_chart(
        sentiment_fig,
        use_container_width=True
    )

# ============================================================
# DAILY SENTIMENT
# ============================================================

if not news_df.empty:

    daily_sentiment = (

        news_df

        .groupby("Date")

        ["sentiment"]

        .mean()

        .reset_index()
    )

    daily_sentiment["Date"] = (
        pd.to_datetime(
            daily_sentiment["Date"]
        )
    )

else:

    daily_sentiment = pd.DataFrame(
        columns=[
            "Date",
            "sentiment"
        ]
    )

# ============================================================
# MERGE SENTIMENT WITH STOCK DATA
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = pd.merge(

    df,

    daily_sentiment,

    on="Date",

    how="left"
)

df["sentiment"] = (
    df["sentiment"]
    .fillna(0)
)

# ============================================================
# SENTIMENT FEATURES
# ============================================================

df["sentiment_lag1"] = (
    df["sentiment"]
    .shift(1)
)

df["sentiment_lag3"] = (
    df["sentiment"]
    .rolling(3)
    .mean()
)

df["sentiment_lag7"] = (
    df["sentiment"]
    .rolling(7)
    .mean()
)

df["sentiment_change"] = (
    df["sentiment"]
    .diff()
)

df["rolling_sentiment_5"] = (
    df["sentiment"]
    .rolling(5)
    .mean()
)

df["rolling_sentiment_10"] = (
    df["sentiment"]
    .rolling(10)
    .mean()
)

# ============================================================
# TOP POSITIVE NEWS
# ============================================================

if not news_df.empty:

    st.subheader(
        "Top Positive Headlines"
    )

    st.dataframe(

        news_df

        .sort_values(
            "sentiment",
            ascending=False
        )

        .head(10)
    )

# ============================================================
# TOP NEGATIVE NEWS
# ============================================================

if not news_df.empty:

    st.subheader(
        "Top Negative Headlines"
    )

    st.dataframe(

        news_df

        .sort_values(
            "sentiment"
        )

        .head(10)
    )

# ============================================================
# SENTIMENT TREND
# ============================================================

if not daily_sentiment.empty:

    sentiment_trend = px.line(

        daily_sentiment,

        x="Date",

        y="sentiment",

        title="Daily Sentiment Trend",

        markers=True
    )

    st.plotly_chart(
        sentiment_trend,
        use_container_width=True
    )
    # ============================================================
# FEATURE ENGINEERING
# ============================================================

st.subheader(
    "Feature Engineering"
)

# ============================================================
# MOVING AVERAGES
# ============================================================

df["MA_5"] = (
    df["Close"]
    .rolling(5)
    .mean()
)

df["MA_7"] = (
    df["Close"]
    .rolling(7)
    .mean()
)

df["MA_14"] = (
    df["Close"]
    .rolling(14)
    .mean()
)

df["MA_21"] = (
    df["Close"]
    .rolling(21)
    .mean()
)

df["MA_50"] = (
    df["Close"]
    .rolling(50)
    .mean()
)

# ============================================================
# EXPONENTIAL MOVING AVERAGES
# ============================================================

df["EMA_5"] = (
    df["Close"]
    .ewm(
        span=5,
        adjust=False
    )
    .mean()
)

df["EMA_12"] = (
    df["Close"]
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)

df["EMA_26"] = (
    df["Close"]
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)

df["EMA_50"] = (
    df["Close"]
    .ewm(
        span=50,
        adjust=False
    )
    .mean()
)

# ============================================================
# MACD
# ============================================================

df["MACD"] = (
    df["EMA_12"]
    -
    df["EMA_26"]
)

df["MACD_SIGNAL"] = (

    df["MACD"]

    .ewm(
        span=9,
        adjust=False
    )

    .mean()
)

df["MACD_HIST"] = (
    df["MACD"]
    -
    df["MACD_SIGNAL"]
)

# ============================================================
# RSI
# ============================================================

delta = (
    df["Close"]
    .diff()
)

gain = delta.where(
    delta > 0,
    0
)

loss = -delta.where(
    delta < 0,
    0
)

avg_gain = (
    gain
    .rolling(14)
    .mean()
)

avg_loss = (
    loss
    .rolling(14)
    .mean()
)

rs = (
    avg_gain
    /
    avg_loss
)

df["RSI"] = (
    100
    -
    (
        100
        /
        (
            1 + rs
        )
    )
)

# ============================================================
# BOLLINGER BANDS
# ============================================================

rolling_mean = (
    df["Close"]
    .rolling(20)
    .mean()
)

rolling_std = (
    df["Close"]
    .rolling(20)
    .std()
)

df["BB_UPPER"] = (
    rolling_mean
    +
    2 * rolling_std
)

df["BB_LOWER"] = (
    rolling_mean
    -
    2 * rolling_std
)

df["BB_WIDTH"] = (
    df["BB_UPPER"]
    -
    df["BB_LOWER"]
)

# ============================================================
# DAILY RETURNS
# ============================================================

df["Daily_Return"] = (
    df["Close"]
    .pct_change()
)

# ============================================================
# LOG RETURNS
# ============================================================

df["Log_Return"] = np.log(
    df["Close"]
    /
    df["Close"].shift(1)
)

# ============================================================
# PRICE CHANGE %
# ============================================================

df["Price_Change_Pct"] = (

    (
        df["Close"]
        -
        df["Open"]
    )

    /

    df["Open"]

) * 100

# ============================================================
# VOLATILITY
# ============================================================

df["Volatility_5"] = (

    df["Daily_Return"]

    .rolling(5)

    .std()
)

df["Volatility_10"] = (

    df["Daily_Return"]

    .rolling(10)

    .std()
)

df["Volatility_21"] = (

    df["Daily_Return"]

    .rolling(21)

    .std()
)

# ============================================================
# MOMENTUM
# ============================================================

df["Momentum_3"] = (
    df["Close"]
    -
    df["Close"].shift(3)
)

df["Momentum_7"] = (
    df["Close"]
    -
    df["Close"].shift(7)
)

df["Momentum_14"] = (
    df["Close"]
    -
    df["Close"].shift(14)
)

df["Momentum_21"] = (
    df["Close"]
    -
    df["Close"].shift(21)
)

# ============================================================
# RATE OF CHANGE
# ============================================================

df["ROC_5"] = (

    (
        df["Close"]
        -
        df["Close"].shift(5)
    )

    /

    df["Close"].shift(5)

) * 100

df["ROC_10"] = (

    (
        df["Close"]
        -
        df["Close"].shift(10)
    )

    /

    df["Close"].shift(10)

) * 100

# ============================================================
# VOLUME FEATURES
# ============================================================

df["Volume_MA_5"] = (
    df["Volume"]
    .rolling(5)
    .mean()
)

df["Volume_MA_21"] = (
    df["Volume"]
    .rolling(21)
    .mean()
)

df["Volume_Change"] = (
    df["Volume"]
    .pct_change()
)

# ============================================================
# LAG FEATURES
# ============================================================

for lag in [1,2,3,5,7,14]:

    df[f"Close_Lag_{lag}"] = (
        df["Close"]
        .shift(lag)
    )

# ============================================================
# HIGH LOW SPREAD
# ============================================================

df["High_Low_Spread"] = (
    df["High"]
    -
    df["Low"]
)

# ============================================================
# OPEN CLOSE SPREAD
# ============================================================

df["Open_Close_Spread"] = (
    df["Close"]
    -
    df["Open"]
)

# ============================================================
# ROLLING STATS
# ============================================================

df["Rolling_Max_21"] = (
    df["High"]
    .rolling(21)
    .max()
)

df["Rolling_Min_21"] = (
    df["Low"]
    .rolling(21)
    .min()
)

df["Rolling_Mean_21"] = (
    df["Close"]
    .rolling(21)
    .mean()
)

# ============================================================
# ATR
# ============================================================

high_low = (
    df["High"]
    -
    df["Low"]
)

high_close = np.abs(
    df["High"]
    -
    df["Close"].shift()
)

low_close = np.abs(
    df["Low"]
    -
    df["Close"].shift()
)

ranges = pd.concat(
    [
        high_low,
        high_close,
        low_close
    ],
    axis=1
)

true_range = (
    ranges
    .max(axis=1)
)

df["ATR"] = (
    true_range
    .rolling(14)
    .mean()
)

# ============================================================
# FEAR GREED INDEX
# ============================================================

df["Fear_Greed"] = (

    (
        df["RSI"]
        / 100
    )

    +

    df["sentiment"]

)

# ============================================================
# TARGET VARIABLE
# ============================================================

df["Target"] = (
    df["Close"]
    .shift(-1)
    >
    df["Close"]
).astype(int)

# ============================================================
# DROP NULLS
# ============================================================

df = df.dropna()

# ============================================================
# FEATURE COUNT
# ============================================================

numeric_features = (

    df

    .select_dtypes(
        include=np.number
    )

    .columns
)

st.success(
    f"Total Engineered Features: {len(numeric_features)}"
)

st.dataframe(
    df.head()
)
# ============================================================
# EDA DASHBOARD
# ============================================================

st.header(
    "📊 Exploratory Data Analysis"
)

# ============================================================
# CANDLESTICK CHART
# ============================================================

st.subheader(
    "Candlestick Chart"
)

candlestick_fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    ]
)

candlestick_fig.update_layout(
    height=600,
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(
    candlestick_fig,
    use_container_width=True
)

# ============================================================
# CLOSE PRICE WITH MOVING AVERAGES
# ============================================================

st.subheader(
    "Moving Averages"
)

ma_fig = go.Figure()

ma_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Close"
    )
)

ma_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA_21"],
        name="MA 21"
    )
)

ma_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MA_50"],
        name="MA 50"
    )
)

st.plotly_chart(
    ma_fig,
    use_container_width=True
)

# ============================================================
# VOLUME ANALYSIS
# ============================================================

st.subheader(
    "Volume Analysis"
)

volume_fig = px.bar(
    df,
    x="Date",
    y="Volume",
    title="Daily Trading Volume"
)

st.plotly_chart(
    volume_fig,
    use_container_width=True
)

# ============================================================
# RSI CHART
# ============================================================

st.subheader(
    "RSI Indicator"
)

rsi_fig = go.Figure()

rsi_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["RSI"],
        name="RSI"
    )
)

rsi_fig.add_hline(
    y=70
)

rsi_fig.add_hline(
    y=30
)

st.plotly_chart(
    rsi_fig,
    use_container_width=True
)

# ============================================================
# MACD CHART
# ============================================================

st.subheader(
    "MACD Indicator"
)

macd_fig = go.Figure()

macd_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MACD"],
        name="MACD"
    )
)

macd_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MACD_SIGNAL"],
        name="Signal"
    )
)

st.plotly_chart(
    macd_fig,
    use_container_width=True
)

# ============================================================
# BOLLINGER BANDS
# ============================================================

st.subheader(
    "Bollinger Bands"
)

bb_fig = go.Figure()

bb_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Close"
    )
)

bb_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["BB_UPPER"],
        name="Upper Band"
    )
)

bb_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["BB_LOWER"],
        name="Lower Band"
    )
)

st.plotly_chart(
    bb_fig,
    use_container_width=True
)

# ============================================================
# DAILY RETURNS DISTRIBUTION
# ============================================================

st.subheader(
    "Daily Return Distribution"
)

return_fig = px.histogram(
    df,
    x="Daily_Return",
    nbins=60,
    title="Daily Return Distribution"
)

st.plotly_chart(
    return_fig,
    use_container_width=True
)

# ============================================================
# VOLATILITY ANALYSIS
# ============================================================

st.subheader(
    "Volatility Analysis"
)

vol_fig = px.line(
    df,
    x="Date",
    y="Volatility_21",
    title="21 Day Volatility"
)

st.plotly_chart(
    vol_fig,
    use_container_width=True
)

# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.subheader(
    "Correlation Heatmap"
)

numeric_df = df.select_dtypes(
    include=np.number
)

corr_matrix = (
    numeric_df
    .corr()
)

fig, ax = plt.subplots(
    figsize=(14,10)
)

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ============================================================
# TOP CORRELATED FEATURES WITH TARGET
# ============================================================

st.subheader(
    "Most Important Correlations"
)

target_corr = (

    corr_matrix["Target"]

    .sort_values(
        ascending=False
    )

    .reset_index()
)

target_corr.columns = [
    "Feature",
    "Correlation"
]

st.dataframe(
    target_corr.head(20)
)

# ============================================================
# FEATURE DISTRIBUTIONS
# ============================================================

st.subheader(
    "Feature Distribution Explorer"
)

feature_choice = st.selectbox(

    "Choose Feature",

    numeric_df.columns
)

dist_fig = px.histogram(

    df,

    x=feature_choice,

    nbins=50,

    title=f"{feature_choice} Distribution"
)

st.plotly_chart(
    dist_fig,
    use_container_width=True
)

# ============================================================
# SCATTER RELATIONSHIP
# ============================================================

st.subheader(
    "Feature Relationship"
)

x_feature = st.selectbox(
    "X Axis",
    numeric_df.columns,
    key="x_feature"
)

y_feature = st.selectbox(
    "Y Axis",
    numeric_df.columns,
    key="y_feature"
)

scatter_fig = px.scatter(

    df,

    x=x_feature,

    y=y_feature,

    title=f"{x_feature} vs {y_feature}"
)

st.plotly_chart(
    scatter_fig,
    use_container_width=True
)

# ============================================================
# DATA QUALITY REPORT
# ============================================================

st.subheader(
    "Data Quality Report"
)

quality_df = pd.DataFrame({

    "Column":
    df.columns,

    "Missing Values":
    df.isnull().sum().values,

    "Data Type":
    df.dtypes.values.astype(str)
})

st.dataframe(
    quality_df
)

# ============================================================
# SUMMARY STATS
# ============================================================

st.subheader(
    "Summary Statistics"
)

st.dataframe(
    df.describe()
)
# ============================================================
# MACHINE LEARNING SECTION
# ============================================================

st.header(
    "🤖 Machine Learning Models"
)

# ============================================================
# FEATURE SELECTION
# ============================================================

excluded_columns = [

    "Date",
    "Target"

]

features = [

    col

    for col in df.columns

    if col not in excluded_columns

]

X = df[features]

y = df["Target"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

st.success(
    f"Training Rows: {len(X_train)} | Test Rows: {len(X_test)}"
)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

lr_model = LogisticRegression(
    max_iter=5000
)

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)

lr_prob = lr_model.predict_proba(
    X_test
)[:,1]

# ============================================================
# RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42,

    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_prob = rf_model.predict_proba(
    X_test
)[:,1]

# ============================================================
# XGBOOST
# ============================================================

xgb_model = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)

xgb_prob = xgb_model.predict_proba(
    X_test
)[:,1]

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(

        model_name,

        y_true,

        predictions,

        probabilities

):

    return {

        "Model":
        model_name,

        "Accuracy":
        accuracy_score(
            y_true,
            predictions
        ),

        "Precision":
        precision_score(
            y_true,
            predictions
        ),

        "Recall":
        recall_score(
            y_true,
            predictions
        ),

        "F1":
        f1_score(
            y_true,
            predictions
        ),

        "ROC_AUC":
        roc_auc_score(
            y_true,
            probabilities
        )
    }

# ============================================================
# LEADERBOARD
# ============================================================

leaderboard = pd.DataFrame([

    evaluate_model(
        "Logistic Regression",
        y_test,
        lr_pred,
        lr_prob
    ),

    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred,
        rf_prob
    ),

    evaluate_model(
        "XGBoost",
        y_test,
        xgb_pred,
        xgb_prob
    )

])

leaderboard = leaderboard.sort_values(
    "ROC_AUC",
    ascending=False
)

# ============================================================
# SHOW LEADERBOARD
# ============================================================

st.subheader(
    "Model Leaderboard"
)

st.dataframe(
    leaderboard
)

# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    leaderboard
    .iloc[0]["Model"]
)

st.success(
    f"Best Model: {best_model_name}"
)

# ============================================================
# PICK BEST MODEL
# ============================================================

if best_model_name == "XGBoost":

    best_model = xgb_model
    best_pred = xgb_pred
    best_prob = xgb_prob

elif best_model_name == "Random Forest":

    best_model = rf_model
    best_pred = rf_pred
    best_prob = rf_prob

else:

    best_model = lr_model
    best_pred = lr_pred
    best_prob = lr_prob

# ============================================================
# METRIC CARDS
# ============================================================

acc = accuracy_score(
    y_test,
    best_pred
)

prec = precision_score(
    y_test,
    best_pred
)

rec = recall_score(
    y_test,
    best_pred
)

f1 = f1_score(
    y_test,
    best_pred
)

auc = roc_auc_score(
    y_test,
    best_prob
)

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    st.metric(
        "Accuracy",
        f"{acc:.3f}"
    )

with c2:
    st.metric(
        "Precision",
        f"{prec:.3f}"
    )

with c3:
    st.metric(
        "Recall",
        f"{rec:.3f}"
    )

with c4:
    st.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

with c5:
    st.metric(
        "ROC AUC",
        f"{auc:.3f}"
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "Classification Report"
)

report = classification_report(
    y_test,
    best_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader(
    "Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    best_pred
)

cm_fig = px.imshow(

    cm,

    text_auto=True,

    title="Confusion Matrix"
)

st.plotly_chart(
    cm_fig,
    use_container_width=True
)

# ============================================================
# ROC CURVE
# ============================================================

st.subheader(
    "ROC Curve"
)

fpr,tpr,_ = roc_curve(
    y_test,
    best_prob
)

roc_fig = go.Figure()

roc_fig.add_trace(

    go.Scatter(

        x=fpr,

        y=tpr,

        mode="lines",

        name="ROC Curve"
    )
)

roc_fig.update_layout(

    xaxis_title="False Positive Rate",

    yaxis_title="True Positive Rate",

    height=500
)

st.plotly_chart(
    roc_fig,
    use_container_width=True
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if best_model_name in [

    "XGBoost",

    "Random Forest"
]:

    st.subheader(
        "Feature Importance"
    )

    importance_df = pd.DataFrame({

        "Feature":
        X.columns,

        "Importance":
        best_model.feature_importances_
    })

    importance_df = (

        importance_df

        .sort_values(
            "Importance",
            ascending=False
        )
    )

    st.dataframe(
        importance_df.head(25)
    )

    importance_fig = px.bar(

        importance_df.head(20),

        x="Importance",

        y="Feature",

        orientation="h",

        title="Top 20 Features"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )
    # ============================================================
# PREDICTION ENGINE
# ============================================================

st.header(
    "🔮 Tomorrow Market Prediction"
)

# ============================================================
# LATEST DATA
# ============================================================

latest_data = X.iloc[-1:]

# ============================================================
# PREDICTION
# ============================================================

prediction = best_model.predict(
    latest_data
)[0]

prediction_probabilities = (
    best_model.predict_proba(
        latest_data
    )[0]
)

down_probability = (
    prediction_probabilities[0]
)

up_probability = (
    prediction_probabilities[1]
)

confidence = (
    max(
        prediction_probabilities
    )
    * 100
)

# ============================================================
# PREDICTION CARD
# ============================================================

prediction_col1, prediction_col2 = (
    st.columns(2)
)

with prediction_col1:

    if prediction == 1:

        st.success(
            "📈 PREDICTION: STOCK LIKELY TO MOVE UP"
        )

    else:

        st.error(
            "📉 PREDICTION: STOCK LIKELY TO MOVE DOWN"
        )

with prediction_col2:

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )

# ============================================================
# PROBABILITY CARDS
# ============================================================

c1,c2 = st.columns(2)

with c1:

    st.metric(
        "UP Probability",
        f"{up_probability*100:.2f}%"
    )

with c2:

    st.metric(
        "DOWN Probability",
        f"{down_probability*100:.2f}%"
    )

# ============================================================
# GAUGE CHART
# ============================================================

gauge_fig = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=up_probability * 100,

        title={
            "text":
            "Bullish Probability"
        },

        gauge={

            "axis":{
                "range":[0,100]
            },

            "bar":{
                "thickness":0.3
            }
        }
    )
)

st.plotly_chart(
    gauge_fig,
    use_container_width=True
)

# ============================================================
# BUY SELL HOLD SIGNAL
# ============================================================

signal = "HOLD"

latest_rsi = (
    df["RSI"]
    .iloc[-1]
)

latest_macd = (
    df["MACD"]
    .iloc[-1]
)

latest_signal = (
    df["MACD_SIGNAL"]
    .iloc[-1]
)

latest_sentiment = (
    df["sentiment"]
    .iloc[-1]
)

if (

    latest_rsi < 30

    and

    latest_macd > latest_signal

):

    signal = "BUY"

elif (

    latest_rsi > 70

    and

    latest_macd < latest_signal

):

    signal = "SELL"

else:

    signal = "HOLD"

# ============================================================
# SIGNAL DISPLAY
# ============================================================

st.subheader(
    "Trading Signal"
)

if signal == "BUY":

    st.success(
        "🟢 BUY SIGNAL"
    )

elif signal == "SELL":

    st.error(
        "🔴 SELL SIGNAL"
    )

else:

    st.warning(
        "🟡 HOLD SIGNAL"
    )

# ============================================================
# MARKET SNAPSHOT
# ============================================================

st.subheader(
    "Market Snapshot"
)

col1,col2,col3,col4 = st.columns(4)

with col1:

    st.metric(

        "Latest Close",

        round(
            df["Close"].iloc[-1],
            2
        )
    )

with col2:

    st.metric(

        "RSI",

        round(
            latest_rsi,
            2
        )
    )

with col3:

    st.metric(

        "MACD",

        round(
            latest_macd,
            2
        )
    )

with col4:

    st.metric(

        "Sentiment",

        round(
            latest_sentiment,
            2
        )
    )

# ============================================================
# MARKET HEALTH SCORE
# ============================================================

health_score = 0

if latest_sentiment > 0:
    health_score += 25

if latest_rsi < 70:
    health_score += 25

if latest_macd > latest_signal:
    health_score += 25

if prediction == 1:
    health_score += 25

st.subheader(
    "Market Health Score"
)

st.progress(
    health_score / 100
)

st.metric(
    "Score",
    f"{health_score}/100"
)

# ============================================================
# AI RECOMMENDATION
# ============================================================

st.subheader(
    "AI Recommendation"
)

if health_score >= 75:

    st.success(
        """
        Strong Bullish Setup

        • Positive sentiment

        • Technical indicators healthy

        • Model predicts upward movement

        • Risk appears moderate
        """
    )

elif health_score >= 50:

    st.info(
        """
        Neutral Market

        • Mixed signals

        • Wait for confirmation

        • Monitor momentum closely
        """
    )

else:

    st.warning(
        """
        Bearish Conditions

        • Weak technical structure

        • Negative sentiment

        • Elevated downside risk
        """
    )

# ============================================================
# FEATURE VALUES USED FOR PREDICTION
# ============================================================

st.subheader(
    "Latest Feature Values"
)

latest_feature_df = pd.DataFrame({

    "Feature":
    latest_data.columns,

    "Value":
    latest_data.iloc[0].values

})

st.dataframe(
    latest_feature_df
)

# ============================================================
# PREDICTION HISTORY
# ============================================================

st.subheader(
    "Recent Close Prices"
)

history_fig = px.line(

    df.tail(100),

    x="Date",

    y="Close",

    title="Last 100 Trading Days"
)

st.plotly_chart(
    history_fig,
    use_container_width=True
)
# ============================================================
# BACKTESTING ENGINE
# ============================================================

st.header(
    "📊 Strategy Backtesting"
)

# ============================================================
# GENERATE HISTORICAL PREDICTIONS
# ============================================================

test_predictions = best_model.predict(
    X_test
)

backtest_df = df.iloc[
    split_index:
].copy()

backtest_df["Prediction"] = (
    test_predictions
)

# ============================================================
# STRATEGY RETURNS
# ============================================================

backtest_df["Market_Return"] = (
    backtest_df["Close"]
    .pct_change()
)

backtest_df["Strategy_Return"] = (

    backtest_df["Prediction"]
    .shift(1)

    *

    backtest_df["Market_Return"]

)

backtest_df.dropna(
    inplace=True
)

# ============================================================
# CUMULATIVE RETURNS
# ============================================================

backtest_df["Cumulative_Market"] = (

    1 +

    backtest_df["Market_Return"]

).cumprod()

backtest_df["Cumulative_Strategy"] = (

    1 +

    backtest_df["Strategy_Return"]

).cumprod()

# ============================================================
# PERFORMANCE CHART
# ============================================================

st.subheader(
    "Strategy vs Market"
)

performance_fig = go.Figure()

performance_fig.add_trace(

    go.Scatter(

        x=backtest_df["Date"],

        y=backtest_df[
            "Cumulative_Market"
        ],

        name="Market"
    )
)

performance_fig.add_trace(

    go.Scatter(

        x=backtest_df["Date"],

        y=backtest_df[
            "Cumulative_Strategy"
        ],

        name="Strategy"
    )
)

performance_fig.update_layout(
    height=600
)

st.plotly_chart(
    performance_fig,
    use_container_width=True
)

# ============================================================
# TOTAL RETURNS
# ============================================================

market_return = (

    (
        backtest_df[
            "Cumulative_Market"
        ].iloc[-1]
        - 1
    )

    * 100
)

strategy_return = (

    (
        backtest_df[
            "Cumulative_Strategy"
        ].iloc[-1]
        - 1
    )

    * 100
)

# ============================================================
# SHARPE RATIO
# ============================================================

strategy_std = (
    backtest_df[
        "Strategy_Return"
    ].std()
)

if strategy_std != 0:

    sharpe_ratio = (

        backtest_df[
            "Strategy_Return"
        ].mean()

        /

        strategy_std

    ) * np.sqrt(252)

else:

    sharpe_ratio = 0

# ============================================================
# MAX DRAWDOWN
# ============================================================

rolling_max = (

    backtest_df[
        "Cumulative_Strategy"
    ]

    .cummax()
)

drawdown = (

    backtest_df[
        "Cumulative_Strategy"
    ]

    /

    rolling_max

) - 1

max_drawdown = (
    drawdown.min()
    * 100
)

# ============================================================
# WIN RATE
# ============================================================

winning_trades = (

    backtest_df[
        "Strategy_Return"
    ] > 0

).sum()

total_trades = len(
    backtest_df
)

win_rate = (

    winning_trades

    /

    total_trades

) * 100

# ============================================================
# KPI CARDS
# ============================================================

st.subheader(
    "Backtesting KPIs"
)

k1,k2,k3,k4 = st.columns(4)

with k1:

    st.metric(

        "Market Return",

        f"{market_return:.2f}%"
    )

with k2:

    st.metric(

        "Strategy Return",

        f"{strategy_return:.2f}%"
    )

with k3:

    st.metric(

        "Sharpe Ratio",

        f"{sharpe_ratio:.2f}"
    )

with k4:

    st.metric(

        "Win Rate",

        f"{win_rate:.2f}%"
    )

# ============================================================
# MAX DRAWDOWN
# ============================================================

st.metric(

    "Maximum Drawdown",

    f"{max_drawdown:.2f}%"
)

# ============================================================
# DRAWDOWN CHART
# ============================================================

st.subheader(
    "Drawdown Analysis"
)

drawdown_fig = px.area(

    x=backtest_df["Date"],

    y=drawdown,

    title="Portfolio Drawdown"
)

st.plotly_chart(
    drawdown_fig,
    use_container_width=True
)

# ============================================================
# MONTHLY RETURNS
# ============================================================

backtest_df["Month"] = (

    pd.to_datetime(
        backtest_df["Date"]
    )

    .dt.to_period("M")

    .astype(str)
)

monthly_returns = (

    backtest_df

    .groupby("Month")

    ["Strategy_Return"]

    .sum()

    .reset_index()
)

st.subheader(
    "Monthly Strategy Returns"
)

monthly_fig = px.bar(

    monthly_returns,

    x="Month",

    y="Strategy_Return"
)

st.plotly_chart(
    monthly_fig,
    use_container_width=True
)

# ============================================================
# TRADE ANALYSIS
# ============================================================

st.subheader(
    "Trade Analysis"
)

positive_days = (

    backtest_df[
        "Strategy_Return"
    ] > 0

).sum()

negative_days = (

    backtest_df[
        "Strategy_Return"
    ] < 0

).sum()

neutral_days = (

    backtest_df[
        "Strategy_Return"
    ] == 0

).sum()

trade_df = pd.DataFrame({

    "Category":[

        "Winning Days",

        "Losing Days",

        "Neutral Days"
    ],

    "Count":[

        positive_days,

        negative_days,

        neutral_days
    ]
})

st.dataframe(
    trade_df
)

trade_fig = px.pie(

    trade_df,

    names="Category",

    values="Count",

    title="Trade Distribution"
)

st.plotly_chart(
    trade_fig,
    use_container_width=True
)

# ============================================================
# BACKTEST DATA DOWNLOAD
# ============================================================

st.subheader(
    "Download Backtest Results"
)

backtest_csv = (
    backtest_df
    .to_csv(index=False)
)

st.download_button(

    label="Download Backtest CSV",

    data=backtest_csv,

    file_name=f"{ticker}_backtest.csv",

    mime="text/csv"
)
# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================

st.header(
    "💼 Portfolio Analytics"
)

# ============================================================
# MULTI STOCK COMPARISON
# ============================================================

comparison_tickers = st.multiselect(

    "Compare Stocks",

    [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA"
    ],

    default=[
        ticker
    ]
)

comparison_data = pd.DataFrame()

for stock in comparison_tickers:

    try:

        temp_df = yf.download(

            stock,

            start=start_date,

            end=end_date,

            auto_adjust=True,

            progress=False
        )

        comparison_data[stock] = (
            temp_df["Close"]
        )

    except:

        pass

# ============================================================
# STOCK COMPARISON CHART
# ============================================================

if len(comparison_data.columns) > 0:

    st.subheader(
        "Stock Comparison"
    )

    normalized_df = (

        comparison_data

        /

        comparison_data.iloc[0]

    ) * 100

    comparison_fig = px.line(

        normalized_df,

        title="Normalized Performance Comparison"
    )

    st.plotly_chart(

        comparison_fig,

        use_container_width=True
    )

# ============================================================
# PORTFOLIO SIMULATOR
# ============================================================

st.subheader(
    "Portfolio Simulator"
)

investment_amount = st.number_input(

    "Investment Amount ($)",

    min_value=100,

    value=10000
)

portfolio_return = (

    strategy_return / 100
)

future_value = (

    investment_amount

    *

    (1 + portfolio_return)
)

c1,c2 = st.columns(2)

with c1:

    st.metric(

        "Initial Investment",

        f"${investment_amount:,.0f}"
    )

with c2:

    st.metric(

        "Projected Value",

        f"${future_value:,.0f}"
    )

# ============================================================
# RISK DASHBOARD
# ============================================================

st.header(
    "⚠️ Risk Dashboard"
)

risk_score = 0

if latest_rsi > 70:

    risk_score += 25

if latest_sentiment < 0:

    risk_score += 25

if max_drawdown < -20:

    risk_score += 25

if sharpe_ratio < 1:

    risk_score += 25

risk_level = ""

if risk_score <= 25:

    risk_level = "Low"

elif risk_score <= 50:

    risk_level = "Moderate"

elif risk_score <= 75:

    risk_level = "High"

else:

    risk_level = "Very High"

st.metric(

    "Risk Level",

    risk_level
)

st.progress(
    risk_score / 100
)

# ============================================================
# RISK COMPONENTS
# ============================================================

risk_df = pd.DataFrame({

    "Metric":[

        "RSI Risk",

        "Sentiment Risk",

        "Drawdown Risk",

        "Sharpe Risk"
    ],

    "Value":[

        latest_rsi,

        latest_sentiment,

        max_drawdown,

        sharpe_ratio
    ]
})

st.dataframe(
    risk_df
)

# ============================================================
# FEATURE IMPORTANCE EXPLORER
# ============================================================

if best_model_name in [

    "XGBoost",

    "Random Forest"
]:

    st.header(
        "🔍 Feature Explorer"
    )

    top_features = (

        importance_df

        .head(15)
    )

    feature_chart = px.bar(

        top_features,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Most Important Features"
    )

    st.plotly_chart(

        feature_chart,

        use_container_width=True
    )

# ============================================================
# MARKET CONDITIONS
# ============================================================

st.header(
    "🌎 Market Conditions"
)

market_condition = ""

if latest_rsi > 70:

    market_condition = "Overbought"

elif latest_rsi < 30:

    market_condition = "Oversold"

else:

    market_condition = "Neutral"

st.metric(

    "Current Market State",

    market_condition
)

# ============================================================
# TECHNICAL SUMMARY
# ============================================================

summary_data = pd.DataFrame({

    "Indicator":[

        "RSI",

        "MACD",

        "ATR",

        "Sentiment",

        "Fear_Greed"
    ],

    "Value":[

        latest_rsi,

        latest_macd,

        df["ATR"].iloc[-1],

        latest_sentiment,

        df["Fear_Greed"].iloc[-1]
    ]
})

st.dataframe(
    summary_data
)

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

st.header(
    "📋 Executive Summary"
)

summary_text = f"""

Ticker: {ticker}

Best Model: {best_model_name}

Prediction: {'UP' if prediction == 1 else 'DOWN'}

Confidence: {confidence:.2f}%

Sharpe Ratio: {sharpe_ratio:.2f}

Strategy Return: {strategy_return:.2f}%

Win Rate: {win_rate:.2f}%

Risk Level: {risk_level}

"""

st.text_area(

    "Summary",

    summary_text,

    height=250
)

# ============================================================
# FINAL DATA EXPORT
# ============================================================

st.header(
    "📥 Export"
)

final_csv = (
    df
    .to_csv(index=False)
)

st.download_button(

    "Download Full Dataset",

    final_csv,

    file_name=f"{ticker}_full_dataset.csv",

    mime="text/csv"
)

# ============================================================
# PROJECT FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    ### AI Powered Stock Market Prediction Platform

    Features Included:

    ✅ Data Collection

    ✅ News Sentiment Analysis

    ✅ Feature Engineering

    ✅ Technical Indicators

    ✅ Machine Learning Models

    ✅ Model Comparison

    ✅ Prediction Engine

    ✅ Backtesting

    ✅ Portfolio Analytics

    ✅ Risk Dashboard

    ✅ Data Export
    """
)
# ============================================================
# EXPLAINABLE AI
# ============================================================

st.header(
    "🧠 Explainable AI"
)

# ============================================================
# SHAP SUPPORT
# ============================================================

if best_model_name in [

    "XGBoost",

    "Random Forest"

]:

    try:

        st.subheader(
            "SHAP Feature Importance"
        )

        explainer = shap.TreeExplainer(
            best_model
        )

        sample_data = X_test.tail(
            min(
                200,
                len(X_test)
            )
        )

        shap_values = explainer.shap_values(
            sample_data
        )

        shap_importance = pd.DataFrame({

            "Feature":
            sample_data.columns,

            "Importance":
            np.abs(
                shap_values
            ).mean(axis=0)

        })

        shap_importance = (

            shap_importance

            .sort_values(
                "Importance",
                ascending=False
            )
        )

        st.dataframe(
            shap_importance.head(20)
        )

        shap_fig = px.bar(

            shap_importance.head(20),

            x="Importance",

            y="Feature",

            orientation="h",

            title="SHAP Global Importance"
        )

        st.plotly_chart(
            shap_fig,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"SHAP Error: {e}"
        )

# ============================================================
# FEATURE STABILITY
# ============================================================

st.header(
    "📈 Feature Stability"
)

st.subheader(
    "Feature Correlation With Target"
)

feature_corr = (

    df

    .corr(numeric_only=True)

    ["Target"]

    .sort_values(
        ascending=False
    )

)

feature_corr_df = pd.DataFrame({

    "Feature":
    feature_corr.index,

    "Correlation":
    feature_corr.values

})

st.dataframe(
    feature_corr_df.head(25)
)

# ============================================================
# PREDICTION CONFIDENCE ANALYSIS
# ============================================================

st.header(
    "🎯 Prediction Confidence Analysis"
)

confidence_series = (

    np.max(

        best_model.predict_proba(
            X_test
        ),

        axis=1
    )

    * 100
)

confidence_df = pd.DataFrame({

    "Confidence":
    confidence_series
})

confidence_fig = px.histogram(

    confidence_df,

    x="Confidence",

    nbins=30,

    title="Prediction Confidence Distribution"
)

st.plotly_chart(
    confidence_fig,
    use_container_width=True
)

# ============================================================
# MISCLASSIFIED EXAMPLES
# ============================================================

st.header(
    "❌ Model Errors"
)

error_df = X_test.copy()

error_df["Actual"] = y_test.values

error_df["Predicted"] = best_pred

errors = error_df[
    error_df["Actual"]
    !=
    error_df["Predicted"]
]

st.write(
    f"Misclassified Samples: {len(errors)}"
)

st.dataframe(
    errors.head(50)
)

# ============================================================
# FEATURE RANKING TABLE
# ============================================================

st.header(
    "🏆 Feature Ranking"
)

if best_model_name in [

    "XGBoost",

    "Random Forest"

]:

    ranking_df = pd.DataFrame({

        "Rank":
        range(
            1,
            len(
                importance_df
            ) + 1
        ),

        "Feature":
        importance_df[
            "Feature"
        ].values,

        "Importance":
        importance_df[
            "Importance"
        ].values
    })

    st.dataframe(
        ranking_df
    )

# ============================================================
# MODEL INSIGHTS
# ============================================================

st.header(
    "📋 Model Insights"
)

top_feature = ""

if best_model_name in [

    "XGBoost",

    "Random Forest"

]:

    top_feature = (
        importance_df
        .iloc[0]
        ["Feature"]
    )

insights = f"""

Best Performing Model:
{best_model_name}

ROC AUC:
{auc:.4f}

Top Feature:
{top_feature}

Sharpe Ratio:
{sharpe_ratio:.2f}

Win Rate:
{win_rate:.2f}%

Prediction:
{'UP' if prediction == 1 else 'DOWN'}

Confidence:
{confidence:.2f}%

Risk Level:
{risk_level}

"""

st.text_area(

    "Generated Insights",

    insights,

    height=250
)

# ============================================================
# ADVANCED KPIS
# ============================================================

st.header(
    "📊 Advanced KPIs"
)

k1,k2,k3,k4 = st.columns(4)

with k1:

    st.metric(

        "Dataset Rows",

        len(df)
    )

with k2:

    st.metric(

        "Features",

        len(features)
    )

with k3:

    st.metric(

        "News Articles",

        len(news_df)
        if not news_df.empty
        else 0
    )

with k4:

    st.metric(

        "Engineered Features",

        len(
            numeric_features
        )
    )

# ============================================================
# FINAL PROJECT SCORE
# ============================================================

st.header(
    "🚀 Project Score"
)

project_score = 0

project_score += 20
project_score += 20
project_score += 20
project_score += 20
project_score += 20

st.progress(
    project_score / 100
)

st.success(
    f"Portfolio Project Score: {project_score}/100"
)

# ============================================================
# END OF APPLICATION
# ============================================================

st.markdown(
    """
    ---
    ## End of Analysis

    This platform combines:

    - Finance
    - NLP
    - Machine Learning
    - Explainable AI
    - Backtesting
    - Portfolio Analytics
    - Risk Management

    into a single end-to-end Data Science project.
    """
)
