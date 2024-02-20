from library import *
from process_data import *
import plotly.subplots as ps


st.markdown(
    """
    <style>
        .main-container {
            display: flex;
            justify-content: center;  /* Căn giữa theo chiều ngang */
        }
        .sidebar {
            width: 20%;
            background-color: "#3498db";      
            padding: 20px;
            color: white;
        }
        
        .content {
            width: 60%;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;  /* Căn giữa theo chiều dọc */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()

@st.cache_data
def missing(df):
    missing = df.isna().sum().reset_index()
    missing.columns = ['features', 'total_missing']
    missing['percent'] = (missing['total_missing'] / len(df)) * 100
    missing.index = missing['features']
    plt.figure(figsize=(12, 6))
    missing['total_missing'].plot(kind='bar')
    plt.title('Missing Values Count', fontsize=20)
    st.pyplot(plt)


@st.cache_data
def plot_pie_bar_chart(df, var, title=""):
    df_dummy = preprocess_data(df)

    # Plot Pie Chart
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

    # Pie Chart
    pie_labels = list(df_dummy[var].value_counts().index)
    pie_values = df[var].value_counts().values
    fig.add_trace(go.Pie(labels=pie_labels, values=pie_values, 
                         textinfo="percent+label", marker=dict(colors=px.colors.sequential.Plasma)),
                         row=1, col=1)

    # Bar Chart
    bar_labels = list(df[var].value_counts().index)
    bar_values = df[var].value_counts().values
    fig.add_trace(go.Bar(y=bar_labels, x=bar_values, 
                         orientation='h', marker=dict(color='green'),
                         text=bar_values, textposition='auto'),  # Automatically position text
                         row=1, col=2)

    # Update Layout
    fig.update_layout(title_text="Case distribution of " + var + title, showlegend=False,
                      height=500, width=950)  # Adjust height and width as needed

    # Show Plot
    st.plotly_chart(fig)

@st.cache_data
def open_adj_close_year(df):
    df_dummy= preprocess_data(df)
    fig = px.scatter(df_dummy, x="Open", y="Adj Close", color="Year", title="Open versus Adj Close by Year")
    fig.update_layout(
            autosize=False,
            width=900,
            height=500
        )
    st.plotly_chart(fig)

@st.cache_data
def plot_correlation(df, selected_columns):
    selected_corr = df[selected_columns].corr().abs()['Adj Close'].sort_values(ascending=False)

    # Create a DataFrame for Plotly Express
    corr_df = pd.DataFrame({'Feature': selected_corr.index, 'Correlation': selected_corr.values})

    fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
     title='Correlation Coefficient of Features with Adj Close (Threshold >0.25)',
       labels={'Correlation': 'Coefficient'})
    fig.update_traces(marker_color='red')

    st.plotly_chart(fig)

@st.cache_data
def low_high_quarter(df):
    df_dummy= preprocess_data(df)
    fig = px.scatter(df_dummy, x="Low", y="High", color="Quarter", title="Low versus High by Quarter")
    fig.update_layout(
        autosize=False,
        width=900,
        height=500
    )
    st.plotly_chart(fig)

@st.cache_data   
def adj_close_vol_month(df):
    df_dummy= preprocess_data(df)
    fig = px.scatter(df_dummy, x="Adj Close", y="Volume", color="Month", title="Adj Close versus Volume by Month")
    fig.update_layout(
        autosize=False,
        width=950,
        height=500
    )

    st.plotly_chart(fig)

@st.cache_data
def high_vol_day(df):
    df_dummy= preprocess_data(df)
    fig = px.scatter(df_dummy, x="High", y="Volume", color="Day", title="High versus Volume by Day")
    fig.update_layout(
        autosize=False,
        width=900,
        height=500
    )
    st.plotly_chart(fig)

@st.cache_data

def plot_group(df, title=""):
  
    label_list = list(df.index)
    values = df.values

    # Create subplots: 1 row, 2 columns with different types
    fig = make_subplots(rows=1, cols=2, subplot_titles=[title, title], specs=[[{'type':'domain'}, {'type':'xy'}]])

    # Pie chart
    fig.add_trace(go.Pie(labels=label_list, values=values, textinfo='label+percent', insidetextorientation='radial'), row=1, col=1)

    # Bar chart
    fig.add_trace(go.Bar(x=values, y=label_list, orientation='h', text=values, textposition='auto',showlegend=False), row=1, col=2)
    fig.update_layout(autosize=False, width=900, height=600)
    st.plotly_chart(fig)

@st.cache_data  
def stacked_bar_plot(df, category_column, time_column):
    custom_colors = ['red', 'green', 'blue', 'orange', 'purple']
    stacked_df = df.groupby([time_column, category_column]).size().unstack().fillna(0)
    total_cases = stacked_df.sum(axis=1) 

    fig = go.Figure()
    for i, col in enumerate(stacked_df.columns):
        percentage = stacked_df[col] / total_cases * 100
        fig.add_trace(go.Bar(
            y=stacked_df.index,
            x=percentage,  
            name=col,
            text=percentage.round(1).astype(str) + '%', 
            textposition='inside',
            orientation='h',
            marker_color=custom_colors[i % len(custom_colors)] 
        ))

    fig.update_layout(
        title=f"{time_column} versus {category_column}",
        xaxis_title="Number of Cases",
        yaxis_title=time_column,
        barmode='stack',
        legend_title=category_column,
        width=950,  # Width in pixels
        height=500,
    )

    st.plotly_chart(fig)
    

@st.cache_data
def catvolume_chart(df, category_column):
    category_counts = df[category_column].value_counts()
    total_entries = len(df)
    category_percentages = category_counts / total_entries * 100

    df_pie = pd.DataFrame(category_percentages).reset_index()
    df_pie.columns = [category_column, 'Percentage']

    # Create Pie Chart
    fig_pie = px.pie(df_pie, names=category_column, values='Percentage',
                     title='Pie Chart of ' + category_column, hole=0.3, color_discrete_sequence=px.colors.sequential.Viridis)

    # Create Bar Chart
    fig_bar = px.bar(df, x=category_column, color_discrete_sequence=px.colors.sequential.Viridis,
                     title='Bar Chart of ' + category_column, labels={'x': category_column, 'count': 'Count'})

    # Create Subplots
    fig = ps.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'bar'}]], subplot_titles=('Pie Chart', 'Bar Chart'), horizontal_spacing=0.1)
    
    # Add Pie Chart to subplot
    fig.add_trace(fig_pie['data'][0], row=1, col=1)
    
    # Add Bar Chart to subplot
    for trace in fig_bar['data']:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(title_text=f'Pie and Bar Chart of {category_column}', width=900)

    # Update legend
    fig.update_traces(showlegend=True, selector=dict(type='bar'), name='Bar Chart')
    fig.update_traces(showlegend=True, selector=dict(type='pie'), name='Pie Chart')

    # Show Subplots
    st.plotly_chart(fig)

@st.cache_data
def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.suptitle("Correlation Matrix (Threshold > 0.1)")
    st.pyplot(plt)

@st.cache_data
def plot_low_high_over_year(df_dummy, year):
    fig = go.Figure()

    # Add trace for Low
    fig.add_trace(go.Scatter(x=df_dummy[df_dummy["Year"] == year].index,
                             y=df_dummy[df_dummy["Year"] == year]["Low"],
                             mode='lines+markers',
                             name='Low',
                             line=dict(color='red', width=5),
                             marker=dict(symbol='circle', size=10, color='red')))

    # Add trace for High
    fig.add_trace(go.Scatter(x=df_dummy[df_dummy["Year"] == year].index,
                             y=df_dummy[df_dummy["Year"] == year]["High"],
                             mode='lines+markers',
                             name='High',
                             line=dict(color='blue', width=5),
                             marker=dict(symbol='circle', size=10, color='blue')))

    # Update layout
    fig.update_layout(title=f"The Low and High over year {year}",
                      xaxis=dict(title='Date', showgrid=False),
                      yaxis=dict(title='Values', showgrid=False),
                      legend=dict(title=dict(text='Legend'), orientation='h', yanchor='bottom', y=2, xanchor='right', x=1),
                      showlegend=True,
                      autosize=False,
                      width=1000,
                      height=500)

    st.plotly_chart(fig)

@st.cache_data
def plot_open_close_over_year(df_dummy, year):
    fig = go.Figure()

    # Add trace for Open
    fig.add_trace(go.Scatter(x=df_dummy[df_dummy["Year"] == year].index,
                             y=df_dummy[df_dummy["Year"] == year]["Open"],
                             mode='lines+markers',
                             name='Open',
                             line=dict(color='red', width=5),
                             marker=dict(symbol='circle', size=10, color='red')))

    # Add trace for Close
    fig.add_trace(go.Scatter(x=df_dummy[df_dummy["Year"] == year].index,
                             y=df_dummy[df_dummy["Year"] == year]["Close"],
                             mode='lines+markers',
                             name='Close',
                             line=dict(color='blue', width=5),
                             marker=dict(symbol='circle', size=10, color='blue')))

    # Update layout
    fig.update_layout(title=f"The Open and Close over year {year}",
                      xaxis=dict(title='Date', showgrid=False),
                      yaxis=dict(title='Values', showgrid=False),
                      legend=dict(title=dict(text='Legend'), orientation='h', yanchor='bottom', y=2, xanchor='right', x=1),
                      showlegend=True,
                      autosize=False,
                      width=1000,
                      height=500)

    st.plotly_chart(fig)
@st.cache_data
def plot_adjclose_close_over_year(df_dummy, year):
        fig = go.Figure()

    # Add trace for Open
        fig.add_trace(go.Scatter(x=df_dummy[df_dummy["Year"] == year].index,
                                y=df_dummy[df_dummy["Year"] == year]["Adj Close"],
                                mode='lines+markers',
                                name='Adj Close',
                                line=dict(color='red', width=5),
                                marker=dict(symbol='circle', size=10, color='red')))

        # Add trace for Close
        fig.add_trace(go.Scatter(x=df_dummy[df_dummy["Year"] == year].index,
                                y=df_dummy[df_dummy["Year"] == year]["Close"],
                                mode='lines+markers',
                                name='Close',
                                line=dict(color='blue', width=5),
                                marker=dict(symbol='circle', size=10, color='blue')))

        # Update layout
        fig.update_layout(title=f"The Adj Close and Close over year {year}",
                        xaxis=dict(title='Date', showgrid=False),
                        yaxis=dict(title='Values', showgrid=False),
                        legend=dict(title=dict(text='Legend'), orientation='h', yanchor='bottom', y=2, xanchor='right', x=1),
                        showlegend=True,
                        autosize=False,
                        width=1000,
                        height=500)

        
        st.plotly_chart(fig)
@st.cache_data   
def compute_daily_returns(df):
    daily_return = (df["Adj Close"] / df["Adj Close"].shift(1)) - 1
    daily_return[0] = 0
    return daily_return

@st.cache_data
def adj_close_daily_year(df):
    df_dummy =preprocess_data(df)
    daily_return = compute_daily_returns(df_dummy)
    df_dummy["daily_returns"] = daily_return
    fig, ax1 = plt.subplots(figsize=(16, 8))
    sns.scatterplot(data=df_dummy, x="Adj Close", y="daily_returns", hue="Year", palette="deep", ax=ax1)
    plt.title("Adj Close versus Daily Return by Year")
    st.pyplot(fig)


@st.cache_data
def vol_daily_quarter(df):
    df_dummy =preprocess_data(df)
    daily_return = compute_daily_returns(df_dummy)
    df_dummy["daily_returns"] = daily_return
    fig, ax1 = plt.subplots(figsize=(16, 8))
    sns.scatterplot(data=df_dummy, x="Volume", y="daily_returns", hue="Quarter", palette="deep", ax=ax1)
    plt.title("Volume versus Daily Return by Quarter")
    st.pyplot(fig)


@st.cache_data
def low_daily_month(df):
    df_dummy =preprocess_data(df)
    daily_return = compute_daily_returns(df_dummy)
    df_dummy["daily_returns"] = daily_return
    fig, ax1 = plt.subplots(figsize=(16, 8))
    sns.scatterplot(data=df_dummy, x="Low", y="daily_returns", hue="Month", palette="deep", ax=ax1)
    plt.title("Low versus Daily Return by Month")
    st.pyplot(fig)

@st.cache_data
def high_daily_day(df):
    df_dummy =preprocess_data(df)
    daily_return = compute_daily_returns(df_dummy)
    df_dummy["daily_returns"] = daily_return
    fig, ax1 = plt.subplots(figsize=(16, 8))
    sns.scatterplot(data=df_dummy, x="High", y="daily_returns", hue="Day", palette="deep", ax=ax1)
    plt.title("High versus Daily Return by Day")
    st.pyplot(fig)




@st.cache_data
def technical_indicator(stock_close, SMA_CLOSE, upper_band, lower_band, DIF, MACD, RSI, STDEV):
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    stock_close[:365].plot(title='GLD Moving Average', label='GLD')
    SMA_CLOSE[:365].plot(label="SMA")
    upper_band[:365].plot(label='upper band')
    lower_band[:365].plot(label='lower band')
    plt.ylabel('Price')
    plt.legend(loc='lower left')

    plt.subplot(2, 2, 2)
    DIF[:365].plot(title='DIF and MACD', label='DIF')
    MACD[:365].plot(label='MACD')
    plt.ylabel('Price')
    plt.legend(loc='lower left')

    plt.subplot(2, 2, 3)
    RSI[:365].plot(title='RSI', label='RSI')
    plt.ylabel('Price')
    plt.legend(loc='lower left')

    plt.subplot(2, 2, 4)
    STDEV[:365].plot(title='STDEV', label='STDEV')
    plt.ylabel('Price')
    plt.legend(loc='lower left')
    plt.tight_layout()


    st.pyplot(plt.gcf())

@st.cache_data
def plot_open_close_high_low(df):
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,10))
    open_close = df['Open'] - df['Adj Close']
    open_close[:365].plot(title='Open-Close', label='Open-Close', ax=axes[0])
    high_low = df['High'] - df['Low']
    high_low[:365].plot(title='High-Low', label='High-Low', color='#3483ba',ax=axes[1])
   
    axes[0].set_ylabel('Price')
    axes[1].set_ylabel('Price')
    axes[0].legend(loc='lower left')
    axes[1].legend(loc='lower left')
    plt.tight_layout()
   
    st.pyplot(fig)


@st.cache_data
def plot_yearly_low_high(df):
    df_dummy = preprocess_data(df)
    df_filtered = preprocess_dataframe(df_dummy)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Year-Wise Mean and EWM of Low", "Year-Wise Mean and EWM of High"), shared_yaxes=False)

    year_data_low = df_filtered["Low"].resample('Y').mean()
    year_data_low_ewm = year_data_low.ewm(span=5).mean()
    fig.add_trace(go.Scatter(x=year_data_low.index.year, y=year_data_low, mode='lines', name='Low Mean'), row=1, col=1)
    fig.add_trace(go.Scatter(x=year_data_low.index.year, y=year_data_low_ewm, mode='lines', name='Low EWM'), row=1, col=1)

    year_data_high = df_filtered["High"].resample('Y').mean()
    year_data_high_ewm = year_data_high.ewm(span=5).mean()
    fig.add_trace(go.Scatter(x=year_data_high.index.year, y=year_data_high, mode='lines', name='High Mean'), row=1, col=2)
    fig.add_trace(go.Scatter(x=year_data_high.index.year, y=year_data_high_ewm, mode='lines', name='High EWM'), row=1, col=2)

    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Low Value", row=1, col=1)
    fig.update_yaxes(title_text="High Value", row=1, col=2)

    fig.update_layout(title="Year-Wise Mean and EWM of Low and High", showlegend=True, legend=dict(font=dict(size=15)),width=900,   height=500)
    st.plotly_chart(fig, width=1200, height=500)

@st.cache_data
def plot_yearly_open_close(df):
    df_dummy = preprocess_data(df)
    df_filtered = preprocess_dataframe(df_dummy)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Year-Wise Mean and EWM of Low", "Year-Wise Mean and EWM of High"), shared_yaxes=False)

    year_data_low = df_filtered["Open"].resample('Y').mean()
    year_data_low_ewm = year_data_low.ewm(span=5).mean()
    fig.add_trace(go.Scatter(x=year_data_low.index.year, y=year_data_low, mode='lines', name='Low Mean'), row=1, col=1)
    fig.add_trace(go.Scatter(x=year_data_low.index.year, y=year_data_low_ewm, mode='lines', name='Low EWM'), row=1, col=1)

    year_data_high = df_filtered["Close"].resample('Y').mean()
    year_data_high_ewm = year_data_high.ewm(span=5).mean()
    fig.add_trace(go.Scatter(x=year_data_high.index.year, y=year_data_high, mode='lines', name='High Mean'), row=1, col=2)
    fig.add_trace(go.Scatter(x=year_data_high.index.year, y=year_data_high_ewm, mode='lines', name='High EWM'), row=1, col=2)

    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Open Value", row=1, col=1)
    fig.update_yaxes(title_text="Close Value", row=1, col=2)

    fig.update_layout(title="Year-Wise Mean and EWM of Open and Close", showlegend=True, legend=dict(font=dict(size=15)),width=900,   height=500)
    st.plotly_chart(fig, width=1200, height=500)

@st.cache_data
def plot_normalized_yearly_data(norm_data):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(data=norm_data, marker='s', linewidth=5, ax=ax)
    ax.grid(True)
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized Year-Wise Data', fontsize=25)

    # Display the plot in Streamlit app
    st.pyplot(fig)
