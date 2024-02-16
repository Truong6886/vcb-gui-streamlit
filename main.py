# main.py
import streamlit as st
st.set_page_config(page_title="Dự báo giá cổ phiếu", page_icon="📈", layout="wide")
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
    
    
   
    
    load_data_flag = False
    missing_flag =  False
    corr_flag = False
    
    if load_data_button:
        st.success("Data load successfully")
        st.dataframe(df, width=1200, height=500)
        load_data_flag = True
        chosen_plot = None
        technical_button = None
        yw_button = None
        forecast_button = None
        
    if corr_button and not load_data_flag:
        corr_flag =True
        chosen_plot = None
        technical_button = None
        yw_button = None
        forecast_button = None
        plot_correlation(df, selected_columns)

    if missing_button and not load_data_flag:
        missing_flag = True
        chosen_plot = None
        yw_button = None
        technical_button = None
        forecast_button = None
        missing(df)
  
    
    chosen_plot = st.sidebar.selectbox("Choose distribution", ["", "Day", "Month", "Quarter", "Year", "Open versus Adj Close versus Year","Low versus High versus Quarter","Adj Close versus Volume versus Month"
                 ,"High versus Volume by Day","The distribution of Volume by Year","The distribution of Volume by Month","The distribution of Volume by Day","The distribution of Volume by Quarter","Year versus Categorized Volume",
                 "Day versus Categorized Volume","Month versus Categorized Volume","Quarter versus Categorized Volume","Categorized Volume","Correlation Matrix"],key="chosen_plot")
    yw_button = st.sidebar.selectbox("Year-Wise Time Series Plot",["","Low and High","Open and Close","Adj Close and Close","Year-Wise Mean and EWM of Low and High","Year-Wise Mean and EWM of Open and Close","Normalized Year-Wise Data"],key="yw_button")

    technical_button = st.sidebar.selectbox("Technical Indicators",["","Adj Close versus Daily Return by Year","Volume versus Daily Return by Quarter","Low versus Daily Return by Month","High versus Daily Return by Day","Technical Indicator","Differences"],key="tech_button")
  

    forecast_button = st.sidebar.selectbox("Choose Forecasting Model", ["", "Linear Regression","Lasso Regression"],key="forecast_button")

    
    model_name = st.sidebar.selectbox("Choose Machine Learning Model", ["","Support Vector Machine (SVC)","Logistic Regression"],key="model_button")
    if chosen_plot and not load_data_flag and not missing_flag and not corr_flag:
    
        if chosen_plot == "Day":
            plot_pie_bar_chart(df_dummy, "Day")
        elif chosen_plot == "Month":
            plot_pie_bar_chart(df_dummy, "Month")
        elif chosen_plot == "Quarter":
            plot_pie_bar_chart(df_dummy, "Quarter")            
        elif chosen_plot == "Year":
            plot_pie_bar_chart(df_dummy, "Year")

        elif chosen_plot == "Open versus Adj Close versus Year" and not load_data_flag and not missing_flag and not corr_flag:
            open_adj_close_year(df)
        elif chosen_plot == "Low versus High versus Quarter" and not load_data_flag and not missing_flag and not corr_flag:
            low_high_quarter(df)
        elif chosen_plot == "Adj Close versus Volume versus Month" and not load_data_flag and not missing_flag and not corr_flag:
            adj_close_vol_month(df)
        elif chosen_plot == "High versus Volume by Day" and not load_data_flag and not missing_flag and not corr_flag:
            high_vol_day(df)
        elif chosen_plot == "The distribution of Volume by Year" and not load_data_flag and not missing_flag and not corr_flag:
            plot_group(df_dummy.groupby('Year')['Volume'].sum(), "The distribution of Volume by Year")
        elif chosen_plot == "The distribution of Volume by Month" and not load_data_flag and not missing_flag and not corr_flag:
            plot_group(df_dummy.groupby('Month')['Volume'].sum(), "The distribution of Volume by Month")
        elif chosen_plot == "The distribution of Volume by Day" and not load_data_flag and not missing_flag and not corr_flag:
            plot_group(df_dummy.groupby('Day')['Volume'].sum(), "The distribution of Volume by Day")
        elif chosen_plot == "The distribution of Volume by Quarter" and not load_data_flag and not missing_flag and not corr_flag: 
            plot_group(df_dummy.groupby('Quarter')['Volume'].sum(), "The distribution of Volume by Quarter")
        elif chosen_plot == "Year versus Categorized Volume" and not load_data_flag and not missing_flag and not corr_flag:
            labels = ['30.000-300.000', '30.0000-1.000.000','1.000.000-3.000.000', '>3.000.000']
            df_dummy["Cat_Volume"] = pd.cut(df_dummy["Volume"], [30000, 300000, 1000000, 3000000,max_volume], labels=labels)
            stacked_bar_plot(df_dummy, 'Cat_Volume', 'Year')
        elif chosen_plot == "Day versus Categorized Volume" and not load_data_flag and not missing_flag and not corr_flag:
            labels = ['30.000-300.000', '30.0000-1.000.000','1.000.000-3.000.000', '>3.000.000']
            df_dummy["Cat_Volume"] = pd.cut(df_dummy["Volume"], [30000, 300000, 1000000, 3000000,max_volume], labels=labels)
            stacked_bar_plot(df_dummy, 'Cat_Volume', 'Day')
        elif chosen_plot == "Month versus Categorized Volume" and not load_data_flag and not missing_flag and not corr_flag:
            labels = ['30.000-300.000', '30.0000-1.000.000','1.000.000-3.000.000', '>3.000.000']
            df_dummy["Cat_Volume"] = pd.cut(df_dummy["Volume"], [30000, 300000, 1000000, 3000000,max_volume], labels=labels)
            stacked_bar_plot(df_dummy, 'Cat_Volume', 'Month')
        elif chosen_plot == "Quarter versus Categorized Volume" and not load_data_flag and not missing_flag and not corr_flag:
            labels = ['30.000-300.000', '30.0000-1.000.000','1.000.000-3.000.000', '>3.000.000']
            df_dummy["Cat_Volume"] = pd.cut(df_dummy["Volume"], [30000, 300000, 1000000, 3000000,max_volume], labels=labels)
            stacked_bar_plot(df_dummy, 'Cat_Volume', 'Quarter')
        elif chosen_plot == "Categorized Volume" and not load_data_flag and not missing_flag and not corr_flag:
            labels = ['30.000-300.000', '30.0000-1.000.000','1.000.000-3.000.000', '>3.000.000']
            df_dummy["Cat_Volume"] = pd.cut(df_dummy["Volume"], [30000, 300000, 1000000, 3000000,max_volume], labels=labels)
            catvolume_chart(df_dummy, 'Cat_Volume')
        else : #chosen_plot == "Correlation Matrix" and not load_data_flag and not missing_flag and not corr_flag:
            plot_correlation_matrix(df)
    elif  yw_button and not load_data_flag and not missing_flag and not corr_flag:
        if yw_button == "Low and High":
            plot_low_high_over_year(df_dummy, 2019)
            plot_low_high_over_year(df_dummy, 2020)
        elif yw_button == "Open and Close":
            plot_open_close_over_year(df_dummy, 2021)
            plot_open_close_over_year(df_dummy, 2022)
        elif yw_button == "Adj Close and Close":
            plot_adjclose_close_over_year(df_dummy, 2022)
            plot_adjclose_close_over_year(df_dummy, 2023)   
        elif yw_button == "Year-Wise Mean and EWM of Low and High":
            plot_yearly_low_high(df_filtered)
        elif yw_button == "Year-Wise Mean and EWM of Open and Close":
            plot_yearly_open_close(df_filtered)
        else: #yw_button =="Normalized Year-Wise Data"
            cols = ["Volume", "High", "Low", "Open", "Close", "Adj Close"]
            year_data = df_filtered.resample('y').mean()
            norm_data = (year_data[cols] - year_data[cols].min()) / (year_data[cols].max() - year_data[cols].min())
            plot_normalized_yearly_data(norm_data)
    
    elif technical_button and not load_data_flag and not missing_flag and not corr_flag:
        if technical_button == "Adj Close versus Daily Return by Year":
            adj_close_daily_year(df)
        elif technical_button == "Volume versus Daily Return by Quarter":
            vol_daily_quarter(df)
        elif technical_button == "Low versus Daily Return by Month":
            low_daily_month(df)
        elif technical_button == "High versus Daily Return by Day":
            high_daily_day(df)
            stock_close = df["Adj Close"]
        elif technical_button == "Technical Indicator":
            technical_indicator(stock_close, SMA_CLOSE, upper_band, lower_band, DIF, MACD, RSI, STDEV)
        else: #technical_button == "Differences"
            plot_open_close_high_low(df)
    elif  forecast_button and not load_data_flag and not missing_flag and not corr_flag :
        if forecast_button == "Linear Regression" : 
            X_final, X_train, X_test, X_val, y_final, y_train, y_test, y_val=load_regression_files()
            perform_linear_regression(X_train, y_train, X_test, y_test, X_val, y_val, X_final, y_final, df_dummy)

        if forecast_button == "Lasso Regression":
            X_final, X_train, X_test, X_val, y_final, y_train, y_test, y_val=load_regression_files()
            perform_lasso_regression(X_train, y_train, X_test, y_test, X_val, y_val, X_final, y_final, df_dummy)

    
    elif model_name and not load_data_flag and not missing_flag and not corr_flag:
        if model_name == "Support Vector Machine (SVC)":
           train_svc("SVC_Model")
         

        else : #model_name == "Logistic Regression":
            train_logistic_regression("Logistic Regression  Model")
        
       
        
if __name__ == "__main__":
  
    main()

