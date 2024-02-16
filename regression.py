#regression.py
from library import*
from process_data import *

@st.cache_data
def perform_regression(_df, _model, X, y, xtrain, ytrain, xtest, ytest, xval, yval, label, feat):
    _df = load_data()
    Trr = []
    Tss = []

    _model.fit(xtrain, ytrain)
    predictions_test = _model.predict(xtest)
    predictions_train = _model.predict(xtrain)
    predictions_val = _model.predict(xval)

    rmse_test = np.sqrt(mean_squared_error(ytest, predictions_test))
    mse_test = mean_squared_error(ytest, predictions_test)
    evs_test = explained_variance_score(ytest, predictions_test)

    st.write(f'RMSE using {label}: {rmse_test}')
    st.write('Mean Square Error:', mse_test)
    

    st.write('ACTUAL: Avg. ' + feat + f': {_df[feat].mean()}')
    st.write('ACTUAL: Median ' + feat + f':{_df[feat].median()}')
    st.write('PREDICTED: Avg. ' + feat + f':{predictions_test.mean()}')
    st.write('PREDICTED: Median ' + feat + f':{np.median(predictions_test)}')

    all_pred = _model.predict(X)
    st.write("Mean Square Error (whole dataset): ", mean_squared_error(y, all_pred))
    
    fig = make_subplots(rows=4, cols=2, subplot_titles=['The actual versus predicted  (Training set): '+label,
                                                        'Actual and Predicted Training Set: '+label,
                                                        'The actual versus predicted  (Test set): '+label,
                                                        'Actual and Predicted Test Set: '+label,
                                                        '',
                                                        'Actual and Predicted Validation Set(90 days forecasting) '+label,
                                                        '','Actual and Predicted All Set '+label])

    # Scatter plot for training set
    fig.add_trace(go.Scatter(x=ytrain.squeeze(), y=predictions_train.squeeze(),
                         mode='markers', line=dict(color='blue'), name='Training Data'), row=1, col=1)

    fig.add_trace(go.Scatter(x=ytrain.squeeze(), y=ytrain.squeeze(),
                            mode='lines', line=dict(color='red', dash='dash'),
                            name='Perfect Prediction'), row=1, col=1)
    fig.add_trace(go.Scatter(x=X.index[:split_idx], y=ytrain.squeeze(),
                            mode='lines', line=dict(color='skyblue', width=3),
                            name='Actual'), row=1, col=2)

    fig.add_trace(go.Scatter(x=X.index[:split_idx], y=predictions_train.squeeze(),
                            mode='lines', line=dict(color='red', width=3),
                            name='Predicted'), row=1, col=2)
    # Scatter plot for test set
    ytest_series = ytest.squeeze()  # Define ytest_series here
    fig.add_trace(go.Scatter(x=ytest_series, y=predictions_test.squeeze(),
                            mode='markers', line=dict(color='green'), name='Test Data'), row=2, col=1)

    fig.add_trace(go.Scatter(x=[ytest_series.min(), ytest_series.max()],
                            y=[ytest_series.min(), ytest_series.max()],
                            mode='lines', line=dict(color='red', dash='dash'),
                            name='Perfect Prediction'), row=2, col=1)

    # Actual values trace
    fig.add_trace(go.Scatter(x=X.index[split_idx:], y=ytest.squeeze(),
                            mode='lines', line=dict(color='blue', width=3),
                            name='Actual'), row=2, col=2)

    # Predicted values trace
    fig.add_trace(go.Scatter(x=X.index[split_idx:], y=predictions_test.squeeze(),
                            mode='lines', line=dict(color='red', width=3),
                            name='Predicted'), row=2, col=2)
    fig.add_trace(go.Scatter(x=yval.index, y=yval.squeeze(),
                            mode='lines', line=dict(color='blue', width=3),
                            name='Actual', showlegend=True), row=3, col=2)

    # Predicted values trace
    fig.add_trace(go.Scatter(x=yval.index, y=predictions_val.squeeze(),
                            mode='lines', line=dict(color='red', width=3),
                            name='Predicted', showlegend=True), row=3, col=2)

    fig.add_trace(go.Scatter(x=X_final.index, y=all_pred.squeeze(),
                            mode='markers', line=dict(color='red'), name='Whole Dataset'), row=4, col=2)

    fig.add_trace(go.Scatter(x=X_final.index, y=y.squeeze(),
                            mode='lines', line=dict(color='blue', dash='dash'),
                            name='Perfect Prediction'), row=4, col=2)



    fig.update_layout(showlegend=True,width=1000,height=800)

    st.write(fig)

df=load_data()
X = df.copy()
y_final = pd.DataFrame(X["Adj Close"])
X = X.drop(["Adj Close"], axis =1)
scaler = MinMaxScaler()
X_minmax_data = scaler.fit_transform(X)
X_final = pd.DataFrame(columns=X.columns,
data=X_minmax_data, index=X.index)
print('Shape of features : ', X_final.shape)
print('Shape of target : ', y_final.shape)
    #Shifts target array to predict the n + 1 samples
n=80
y_final = y_final.shift(-1)
y_val = y_final[-n:-1]
y_final = y_final[:-n]
    #Takes last n rows of data to be validation set
X_val = X_final[-n:-1]

X_final = X_final[:-n]
y_final=y_final.astype('float64')


split_idx=round(0.8*len(X))
print("split_idx=",split_idx)
X_train = X_final[:split_idx]
y_train = y_final[:split_idx]
X_test = X_final[split_idx:]
y_test = y_final[split_idx:]
#     # Saves into pkl files
joblib.dump(X, 'regression_pkl/X_Ori.pkl')
joblib.dump(X_final, 'regression_pkl/X_final_reg.pkl')
joblib.dump(X_train, 'regression_pkl/X_train_reg.pkl')
joblib.dump(X_test, 'regression_pkl/X_test_reg.pkl')
joblib.dump(X_val, 'regression_pkl/X_val_reg.pkl')
joblib.dump(y_final, 'regression_pkl/y_final_reg.pkl')
joblib.dump(y_train, 'regression_pkl/y_train_reg.pkl')
joblib.dump(y_test, 'regression_pkl/y_test_reg.pkl')
joblib.dump(y_val, 'regression_pkl/y_val_reg.pkl')
   


def load_regression_files():
    X_Ori = joblib.load('regression_pkl/X_Ori.pkl')
    X_final = joblib.load('regression_pkl/X_final_reg.pkl')
    X_train = joblib.load('regression_pkl/X_train_reg.pkl')
    X_test = joblib.load('regression_pkl/X_test_reg.pkl')
    X_val = joblib.load('regression_pkl/X_val_reg.pkl')
    y_final = joblib.load('regression_pkl/y_final_reg.pkl')
    y_train = joblib.load('regression_pkl/y_train_reg.pkl')
    y_test = joblib.load('regression_pkl/y_test_reg.pkl')  
    y_val = joblib.load('regression_pkl/y_val_reg.pkl') 

    return X_Ori, X_final, X_train, X_test, X_val, y_final, y_train, y_test, y_val 



def perform_linear_regression(X_train, y_train, X_test, y_test, X_val, y_val, X_final, y_final, df_dummy):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    coefficients = lin_reg.coef_
    intercept = lin_reg.intercept_
    st.write("Hệ số β1:", coefficients)
    st.write("Hệ số β0 (Intercept):", intercept)

    y_train_pred = lin_reg.predict(X_train)
    r_squared_train = r2_score(y_train, y_train_pred)
    st.write(f'Hệ số R^2 cho dữ liệu huấn luyện:{r_squared_train}')
    y_test_pred = lin_reg.predict(X_test)
    r_squared_test = r2_score(y_test, y_test_pred)
    st.write(f'Hệ số R^2 cho dữ liệu kiểm tra:{r_squared_test}')

    perform_regression(df_dummy, lin_reg, X_final, y_final, X_train, y_train, X_test, y_test, X_val, y_val, "Linear Regression", "Adj Close")

def perform_decision_tree_regression(X_train, y_train, X_test, y_test, X_val, y_val, X_final, y_final, df_dummy):
    dt_reg = DecisionTreeRegressor(random_state=100)
    dt_reg.fit(X_train, y_train)

    y_train_pred = dt_reg.predict(X_train)
    r_squared_train = r2_score(y_train, y_train_pred)
    st.write(f'Hệ số R^2 cho dữ liệu huấn luyện: {r_squared_train}')

    y_test_pred = dt_reg.predict(X_test)
    r_squared_test = r2_score(y_test, y_test_pred)
    st.write(f'Hệ số R^2 cho dữ liệu kiểm tra: {r_squared_test}')

    perform_regression(df_dummy, dt_reg, X_final, y_final, X_train, y_train, X_test, y_test, X_val, y_val, "DT Regression",  "Adj Close")


def perform_lasso_regression(X_train, y_train, X_test, y_test, X_val, y_val, X_final, y_final, df_dummy):
    alpha = 0.01  # You can adjust the regularization strength

    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train, y_train)

    coefficients = lasso_reg.coef_
    intercept = lasso_reg.intercept_
    st.write("Hệ số β1:", coefficients)
    st.write("Hệ số β0 (Intercept):", intercept)

    y_train_pred = lasso_reg.predict(X_train)
    r_squared_train = r2_score(y_train, y_train_pred)
    st.write(f'Hệ số R^2 cho dữ liệu huấn luyện: {r_squared_train}')

    y_test_pred = lasso_reg.predict(X_test)
    r_squared_test = r2_score(y_test, y_test_pred)
    st.write(f'Hệ số R^2 cho dữ liệu kiểm tra: {r_squared_test}')

    perform_regression(df_dummy, lasso_reg, X_final, y_final, X_train, y_train, X_test, y_test, X_val, y_val, "Lasso Regression", "Adj Close")
# #lin_reg = LinearRegression()
# #lin_reg.fit(X_train, y_train)
# df =load_data()
# #X_final, y_final, X_train, y_train, X_test, y_test, X_val, y_val = perform_regression(df, X, y, X_train, y_train, X_test,y_test, X_val, y_val, label, feat)
# perform_regression(df, lin_reg, X_final, y_final, X_train, y_train, X_test, y_test, X_val, y_val, "Linear Regression", "Adj Close")

# Assuming that '29-12-2023' is a valid date format in your dataset


# Assuming you want to save the plot as an image
