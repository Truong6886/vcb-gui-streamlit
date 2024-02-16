from library import *
from process_data import *

def compute_daily_returns(df):
    daily_return = (df / df.shift(1)) - 1
    daily_return[0] = 0
    return daily_return

def load_and_preprocess_data():
    ticker_symbol = "VCB.VN"
    start_date ="2009-06-30"
    end_date ="2023-12-31"
    data= yf.download(ticker_symbol,start=start_date,end=end_date)
    
    data["daily_returns"] = compute_daily_returns(data["Adj Close"])
    print(data.isnull().sum().to_string())
    print('Total number of null values: ', data.isnull().sum())

    cols = list(data.columns)
    for n in cols:
        data[n].fillna(data[n].mean(), inplace=True)

    print(data.isnull().sum().to_string())
    print("Data Columns --> ", list(data.columns))
    print(data.head(10).to_string())

    return data

data = load_and_preprocess_data()
y = data["daily_returns"]

# Ensures y is of integer type
y = np.array([1 if i > 0 else 0 for i in y]).astype(int)

# Drops irrelevant column
X = data.drop(["daily_returns"], axis=1)

# Checks null values because of technical indicators
print(X.isnull().sum().to_string())
print('Total number of null values: ', X.isnull().sum().sum())

# Fills each null value in every column with mean value
cols = list(X.columns)
for n in cols:
    X[n].fillna(X[n].mean(), inplace=True)

# Checks again null values
print(X.isnull().sum().to_string())
print('Total number of null values: ', X.isnull().sum().sum())

# Check and convert data types
X = X.astype(float)
y = y.astype(int)

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# Splits the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021, stratify=y)

# Use Standard Scaler
scaler = StandardScaler()
X_train_stand = scaler.fit_transform(X_train)
X_test_stand = scaler.transform(X_test)

# Save into pkl files
joblib.dump(X_train_stand, 'machine_learning_pkl/X_train.pkl')
joblib.dump(X_test_stand, 'machine_learning_pkl/X_test.pkl')
joblib.dump(y_train, 'machine_learning_pkl/y_train.pkl')
joblib.dump(y_test, 'machine_learning_pkl/y_test.pkl')

def load_files():
    X_train = joblib.load('machine_learning_pkl/X_train.pkl')
    X_test = joblib.load('machine_learning_pkl/X_test.pkl')
    y_train = joblib.load('machine_learning_pkl/y_train.pkl')
    y_test = joblib.load('machine_learning_pkl/y_test.pkl')
    return X_train, X_test, y_train, y_test

def plot_learning_curve(estimator, name, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(figsize=(10, 5))
    elif isinstance(axes, np.ndarray):  # Check if axes is an ndarray
        axes = axes[0]  # Take the first element of the array
    axes.set_title(name + " with " )
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")
    plt.show()

    return plt

def plot_real_pred_val(Y_test, ypred, name,ax):
    acc=accuracy_score(Y_test,ypred)
    ax.scatter(range(len(ypred)),ypred,color="blue",\
    lw=5,label="Predicted")
    ax.scatter(range(len(Y_test)), \
    Y_test,color="red",label="Actual")
    ax.set_title("Predicted Values vs True Values of " + name + " with " )
    ax.set_xlabel("Accuracy: " + str(round((acc*100),3)) + "%")
    ax.set_ylabel("Daily Returns")
    ax.legend()
    ax.grid(True, alpha=0.75, lw=1, ls='-.')
    ax.yaxis.set_ticklabels(["", "Negative Daily Returns", "", "", "", "",
    "Positive Daily Returns"]);
#plt.show()

def plot_cm(Y_test, ypred, name,ax):
    cm = confusion_matrix(Y_test, ypred)
    sns.heatmap(cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', cmap="Greens", ax=ax)
    ax.set_title(name + ' Confusion Matrix ' + "with " )
    ax.set_xlabel('Y predict')
    ax.set_ylabel('Y test')
    ax.xaxis.set_ticklabels(["Negative Daily Returns", "Positive Daily Returns"]);
    ax.yaxis.set_ticklabels(["Negative Daily Returns", "Positive Daily Returns"]);
    #plt.show()


#Plots ROC

def plot_roc(model,X_test, y_test, name,ax):
    Y_pred_prob = model.predict_proba(X_test)
    Y_pred_prob = Y_pred_prob[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, Y_pred_prob)
    ax.plot([0,1],[0,1], color='navy', lw=5, linestyle='--')
    ax.plot(fpr,tpr, label='ANN')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve of ' + name + " with ")
    plt.grid(True)
#plt.show()
#Chooses two features for decision boundary
X_feature = X.iloc[:, 13:14]
X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(X_feature, y, test_size = 0.2, random_state = 42, stratify=y)
def train_model(model, X, y):
    model.fit(X, y)
    return model
def predict_model(model, X, proba=False):
        if ~proba:
            y_pred = model.predict(X)
        else:
            y_pred_proba = model.predict_proba(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred
list_scores = []



def train_svc(model_name):
    
    X_train, X_test, y_train, y_test = load_files()
    decimal_places = 2

    # Train the model
    model_svc = SVC(C=10, gamma=0.1, kernel='linear', probability=True, random_state=2021)
    _model = train_model(model_svc, X_train, y_train)
    
    # Make predictions on the test set
    y_pred = predict_model(_model, X_test, proba=False)

  
    
    
    # Print the prediction and the actual last closing price
    
    # Model evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Round the percentage values
    accuracy_rounded = round(accuracy * 100, decimal_places)
    recall_rounded = round(recall * 100, decimal_places)
    precision_rounded = round(precision * 100, decimal_places)
    f1_rounded = round(f1 * 100, decimal_places)

    # Display evaluation metrics in Streamlit
    st.write("Accuracy:", f"{accuracy_rounded:.{decimal_places}f}%")
    st.write("Recall:", f"{recall_rounded:.{decimal_places}f}%")
    st.write("Precision:", f"{precision_rounded:.{decimal_places}f}%")
    st.write("F1 Score:", f"{f1_rounded:.{decimal_places}f}%")

    # Display result table with an expander
    result_table = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    result_table['Result'] = result_table['Actual'] == result_table['Predicted']
    
    with st.expander("Result", expanded=False):
        st.table(result_table.style.applymap(lambda x: 'background-color: #00FF00' if x else 'background-color: #FF0000', subset=['Result']))

    # Plotting evaluation metrics in subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].set_title("Confusion Matrix")
    plot_cm(y_test, y_pred, model_name, axes[0, 0])
    axes[0, 1].set_title("Real vs. Predicted Values")
    plot_real_pred_val(y_test, y_pred, model_name, axes[0, 1])
    axes[1, 0].set_title("ROC Curve")
    plot_roc(_model, X_test, y_test, model_name, axes[1, 0])
    axes[1, 1].set_title("Learning Curve")
    plot_learning_curve(_model, model_name, X_train, y_train, axes=axes[1, 1])
    
    # X_train, y_train, X_test, y_test,split = create_train_test_splits(vcb_adj)
    # vcb_adj = prepare_stock_data(df)
    # plot_predictions(model, X_test, y_test,df,split ,vcb_adj)
    # Adjust layout and display the plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)



def train_logistic_regression(model_name):
    X_train, X_test, y_train, y_test = load_files()
    decimal_places = 2

    # Define the parameter grid for GridSearchCV
    

    # Create a Logistic Regression model
    logreg = LogisticRegression(max_iter=5000, random_state=42,C=0.01,penalty= 'none', solver='newton-cg')

    # Perform GridSearchCV
 

    # Train the model
    _model = train_model(logreg, X_train, y_train)

    # Make predictions on the test set
    y_pred = predict_model(_model, X_test, proba=False)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Round the percentage values
    accuracy_rounded = round(accuracy * 100, decimal_places)
    recall_rounded = round(recall * 100, decimal_places)
    precision_rounded = round(precision * 100, decimal_places)
    f1_rounded = round(f1 * 100, decimal_places)

  
    st.write("Accuracy:", f"{accuracy_rounded:.{decimal_places}f}%")
    st.write("Recall:", f"{recall_rounded:.{decimal_places}f}%")
    st.write("Precision:", f"{precision_rounded:.{decimal_places}f}%")
    st.write("F1 Score:", f"{f1_rounded:.{decimal_places}f}%")

    # Plot various metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].set_title("Confusion Matrix")
    plot_cm(y_test, y_pred, model_name, axes[0, 0])
    axes[0, 1].set_title("Real vs. Predicted Values")
    plot_real_pred_val(y_test, y_pred, model_name, axes[0, 1])
    axes[1, 0].set_title("ROC Curve")
    plot_roc(_model, X_test, y_test, model_name, axes[1, 0])
    axes[1, 1].set_title("Learning Curve")
    plot_learning_curve(_model, model_name, X_train, y_train, axes=axes[1, 1])
    plt.tight_layout()
    st.pyplot(fig)

# def train_dt(model_name):
#         decimal_places = 2   
#         dt = DecisionTreeClassifier(criterion="entropy",max_depth=48,min_samples_leaf=1,min_samples_split=2,random_state=2021)
#         _model = train_model(dt, X_train, y_train)
#         y_pred = predict_model(_model, X_test, proba=False)
#         accuracy = accuracy_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred, average='weighted')
#         precision = precision_score(y_test, y_pred, average='weighted')
#         f1 = f1_score(y_test, y_pred, average='weighted')

#         accuracy_rounded = round(accuracy * 100, decimal_places)
#         recall_rounded = round(recall * 100, decimal_places)
#         precision_rounded = round(precision * 100, decimal_places)
#         f1_rounded = round(f1 * 100, decimal_places)

#         st.write("Accuracy:", f"{accuracy_rounded:.{decimal_places}f}%")
#         st.write("Recall:", f"{recall_rounded:.{decimal_places}f}%")
#         st.write("Precision:", f"{precision_rounded:.{decimal_places}f}%")
#         st.write("F1 Score:", f"{f1_rounded:.{decimal_places}f}%")

#         fig, axes = plt.subplots(2, 2, figsize=(12, 9))
#         axes[0, 0].set_title("Confusion Matrix")
#         plot_cm(y_test, y_pred, model_name, axes[0, 0])
#         axes[0, 1].set_title("Real vs. Predicted Values")
#         plot_real_pred_val(y_test, y_pred, model_name, axes[0, 1])
#         axes[1, 0].set_title("ROC Curve")
#         plot_roc(_model, X_test, y_test, model_name, axes[1, 0])
#         axes[1, 1].set_title("Learning Curve")
#         plot_learning_curve(_model, model_name, X_train, y_train, axes=axes[1, 1])
#         plt.tight_layout()
#         st.pyplot(fig)

