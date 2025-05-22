import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import datetime
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("Warehouse.csv")

pd.set_option('display.max_columns', None)

data.head()

#Check for missing values
data.isnull().sum()


#To check Duplicates
data[data.duplicated()]


# To show Outliers in the data set run the code 

num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(25, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()


# To remove outliers use this function

def zohaib (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["RETAIL SALES"])

data = zohaib(data,"RETAIL SALES")

#EDA automate

from autoviz.AutoViz_Class import AutoViz_Class 
AV = AutoViz_Class()
import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'Warehouse.csv'
sep =","
dft = AV.AutoViz(
    filename  
)


data["TOTAL_SALES"] = data["RETAIL SALES"] + data["RETAIL TRANSFERS"] + data["WAREHOUSE SALES"]

data["YEAR"] = data["YEAR"].astype(int)
data["MONTH"] = data["MONTH"].astype(int)


monthly_data = data.groupby(["YEAR", "MONTH", "ITEM CODE", "ITEM DESCRIPTION", "ITEM TYPE"]).agg({
    "TOTAL_SALES": "sum",
    "RETAIL SALES": "sum",
    "RETAIL TRANSFERS": "sum",
    "WAREHOUSE SALES": "sum"
}).reset_index()


monthly_data["DATE"] = pd.to_datetime(monthly_data["YEAR"].astype(str) + "-" + monthly_data["MONTH"].astype(str) + "-01")

encoded_data = pd.get_dummies(monthly_data, columns=["ITEM TYPE"])


features = [col for col in encoded_data.columns if col not in ['TOTAL_SALES', 'DATE', 'ITEM DESCRIPTION', 'ITEM CODE']]
X = encoded_data[features]
y = encoded_data["TOTAL_SALES"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

explainer = shap.Explainer(model)
shap_values = explainer(X_test)


shap.summary_plot(shap_values, X_test, plot_type="bar")


plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label="Actual")
plt.plot(y_pred[:50], label="Predicted", linestyle="--")
plt.title("Actual vs Predicted Sales (sample 50)")
plt.legend()
plt.show()
