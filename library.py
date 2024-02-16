import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.linear_model import Lasso
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,precision_score
from sklearn.metrics import f1_score,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score, explained_variance_score,accuracy_score,roc_curve