import streamlit as st
from library import *
from process_data import *
from plot import *
from regression import *
from machine_learning import *

def main():
    df = load_data()
   
    df_dummy = preprocess_data(df)
    df_filtered=preprocess_dataframe(df_dummy)
    max_volume = df_dummy['Volume'].max()
    selected_columns = ['High', 'Low', 'Close', 'Adj Close', 'Volume']
    load_data_button = st.sidebar.button("Load Data")
    missing_button = st.sidebar.button("Missing Values")
    corr_button = st.sidebar.button("Correlation Coefficient")
    stock_close = df["Adj Close"]
    SMA_CLOSE = calculate_SMA(stock_close)
    upper_band, lower_band = calculate_BB(stock_close)
    DIF, MACD = calculate_MACD(stock_close)
    RSI = calculate_RSI(stock_close)
    STDEV = calculate_stdev(stock_close)
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox("Go to", ["Home", "About","j"])

    if page == "Home":
        st.success("Data load successfully")
        st.dataframe(df, width=1200, height=500)

    elif page == "About":
        st.sidebar.selectbox("", ["fsfs", "About"])
        # st.title("About Page")
        # st.write("Welcome to the About Page!")
        # with st.sidebar.expander("Expand About"):
        #     st.write("This is more information about the About page.")
        #     st.write("This is also part of the About page.")
        # with st.sidebar.expander("Expand FDSFE"):
        #     st.write("This is more information about the FDSFE page.")
    elif page == "j":
         st.sidebar.selectbox("", ["fad", "fegregt"])
    page2 = st.sidebar.selectbox("Go to", ["hgjhrg", "gsádfsàdf","âfsàdád"])
    if page2 == "hgjhrg":
        st.success("Data load successfully")
        st.dataframe(df, width=1200, height=500)

    elif page2 == "gsádfsàdf":
        st.sidebar.selectbox("", ["fsfs", "About"])
        # st.title("About Page")
        # st.write("Welcome to the About Page!")
        # with st.sidebar.expander("Expand About"):
        #     st.write("This is more information about the About page.")
        #     st.write("This is also part of the About page.")
        # with st.sidebar.expander("Expand FDSFE"):
        #     st.write("This is more information about the FDSFE page.")
    elif page2 == "âfsàdád":
         st.sidebar.selectbox("", ["fad", "fegregt"])
if __name__ == "__main__":
    main()
