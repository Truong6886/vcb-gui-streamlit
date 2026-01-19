import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

@st.cache_data
def load_data():
    df = stock_historical_data(
        symbol="VCB",
        start_date="2009-06-30",
        end_date="2024-06-04"
    )

    if df.empty:
        st.error("❌ Không tải được dữ liệu cổ phiếu VCB.")
        st.stop()

    df = df.set_index("time")
    return df




@st.cache_data
def preprocess_data(df):
    df.index = pd.to_datetime(df.index)
    df['Day'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week
    df['Quarter'] = df.index.quarter
    
    df_dummy = df.copy()
    #  Ánh xạ giá trị số thành các chuỗi có ý nghĩa
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    df_dummy['Day'] = df_dummy['Day'].map(days)
    months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    df_dummy['Month'] = df_dummy['Month'].map(months)
    quarters = {1: 'Jan-March', 2: 'April-June', 3: 'July-Sept', 4: 'Oct-Dec'}
    df_dummy['Quarter'] = df_dummy['Quarter'].map(quarters)
    return df_dummy


@st.cache_data
def calculate_RSI(df, periods=14):
# wilder's RSI
    delta = df.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rUp = up.ewm(com=periods,adjust=False).mean()
    rDown = down.ewm(com=periods, adjust=False).mean().abs()
    rsi = 100 - 100 / (1 + rUp / rDown)
    return rsi

@st.cache_data
def calculate_SMA(df, peroids=15):
    SMA = df.rolling(window=peroids, min_periods=peroids,
    center=False).mean()
    return SMA

@st.cache_data
def calculate_BB(df, peroids=15):
    STD = df.rolling(window=peroids,min_periods=peroids, center=False).std()
    SMA = calculate_SMA(df)
    upper_band = SMA + (2 * STD)
    lower_band = SMA - (2 * STD)
    return upper_band, lower_band

@st.cache_data
def calculate_stdev(df,periods=5):
    STDEV = df.rolling(periods).std()
    return STDEV

@st.cache_data
def calculate_MACD(df, nslow=26, nfast=12):
    emaslow = df.ewm(span=nslow, min_periods=nslow, adjust=True, ignore_na=False).mean()
    emafast = df.ewm(span=nfast, min_periods=nfast, adjust=True, ignore_na=False).mean()
    dif = emafast - emaslow
    MACD = dif.ewm(span=9, min_periods=9, adjust=True, ignore_na=False).mean()
    return dif, MACD

def preprocess_dataframe(df):
    # Exclude specified columns
    df_dummy =preprocess_data(df)
    df_filtered = df_dummy.copy()

    # Map weekdays to numbers
    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
    }
    df_filtered['Day'] = df_filtered['Day'].map(weekday_mapping)

    # Map months to numbers
    month_mapping = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }
    df_filtered['Month'] = df_filtered['Month'].map(month_mapping)

    # Map quarters to numbers
    quarter_mapping = {
        'July-Sept': 3,  # Assuming July-Sept represents Q3
        'Oct-Dec': 4,    # Assuming Oct-Dec represents Q4
    }
    df_filtered['Quarter'] = df_filtered['Quarter'].map(quarter_mapping)

    return df_filtered
