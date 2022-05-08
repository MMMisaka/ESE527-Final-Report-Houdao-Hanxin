# -------------------------------------------Import-------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.metrics import r2_score
from scipy import stats
pd.set_option("display.max_columns",None)


# -------------------------------------------Notes-------------------------------------------
# The code is better running in jupyternotebook
# The code only contains part of the project including iqr outlier detection and ANN
# You can find more in our R codes


# -------------------------------------------Data Preprocess-------------------------------------------
# Read Data
df = pd.read_csv ('./remake/train.csv')

# Data Asset
# print(df.head())
summary=df.describe()
print(summary)

# Find NullValue
# print(df.isnull().sum().sort_values(ascending=False))

# Data Clean
# 1) Label the categorical data
numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number]).copy()
categorical_data["BsmtQual"]=categorical_data["BsmtQual"].fillna("missing")
categorical_data["BsmtCond"]=categorical_data["BsmtCond"].fillna("missing")
categorical_data["GarageQual"]=categorical_data["GarageQual"].fillna("missing")
categorical_data["GarageCond"]=categorical_data["GarageCond"].fillna("missing")
d={"Ex":3,"Gd":2,"TA":1,"Fa":0,"Po":-1,"missing":-2}
categorical_data["ExterQual"]=categorical_data["ExterQual"].map(d)
categorical_data["ExterCond"]=categorical_data["ExterCond"].map(d)
categorical_data["BsmtQual"]=categorical_data["BsmtQual"].map(d)
categorical_data["BsmtCond"]=categorical_data["BsmtCond"].map(d)
categorical_data["GarageQual"]=categorical_data["GarageQual"].map(d)
categorical_data["GarageCond"]=categorical_data["GarageCond"].map(d)
categorical_data["HeatingQC"]=categorical_data["HeatingQC"].map(d)
categorical_data["KitchenQual"]=categorical_data["KitchenQual"].map(d)
# print(categorical_data[["ExterQual","ExterCond","BsmtQual","BsmtCond","GarageQual","GarageCond","HeatingQC","KitchenQual"]].isnull().sum())
# print(categorical_data[["ExterQual","ExterCond","BsmtQual","BsmtCond","GarageQual","GarageCond","HeatingQC","KitchenQual"]])
# 2) Replace Null with 0 for numerical data
proc_num=numeric_data.fillna(0)
# print(proc_num.info())

# Feature Selection
# More in R codes, here we only draw the heat map
plt.figure(figsize=(30,25))
cor = proc_num.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)


# Check Distribution of Selected Values and target value
data_select= pd.concat([ proc_num[["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
                      "BedroomAbvGr","FullBath","KitchenAbvGr","TotRmsAbvGrd","GarageArea","MiscVal","SalePrice"]],categorical_data[["ExterQual","ExterCond","BsmtQual","BsmtCond","GarageQual","GarageCond","HeatingQC","KitchenQual"]]],axis=1)
plt.figure(figsize=(9,6))
sns.histplot(x="SalePrice",data=data_select)
plt.show()

data_select[["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
                      "BedroomAbvGr","FullBath","KitchenAbvGr","TotRmsAbvGrd","GarageArea","MiscVal","SalePrice"]].plot(kind="box",layout=(3,5),subplots=True,figsize=(10,6))
plt.show()


# -------------------------------------------outlier detection-------------------------------------------
# 1） IsolationForest
# clf = IsolationForest(max_samples=800, contamination="auto")
# clf.fit(x_train)
# y_pred_train = clf.predict(x_train)
# # y_pred_test = clf.predict(x_outliers)
# print(y_pred_train)

# 2) LOF
# clf = LOF(n_neighbors=2)
# res = clf.fit_predict(x)
# print(res)
# print(clf.negative_outlier_factor_)

# 3) IQR Method
outliers={}
for col in ["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
                      "BedroomAbvGr","FullBath","KitchenAbvGr","TotRmsAbvGrd","GarageArea","MiscVal"]:
    q1= data_select[col].quantile(0.25)
    q3= data_select[col].quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    outliers[col]=[lower,upper]
# print(outliers)

# Replace Outliers with Upper and Lower Quantiles
for col in ["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
                      "BedroomAbvGr","FullBath","KitchenAbvGr","TotRmsAbvGrd","GarageArea","MiscVal"]:
    lower,upper=outliers[col]
    data_select[col]=data_select[col].map(lambda x:lower if x<lower else x)
    data_select[col]=data_select[col].map(lambda x:upper if x>upper else x)

# Check Distribution Again
data_select[["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
                      "BedroomAbvGr","FullBath","TotRmsAbvGrd","GarageArea"]].plot(kind="box",layout=(3,5),subplots=True,figsize=(15,6))
plt.show()


# -------------------------------------------Spliting Data-------------------------------------------
x=data_select[["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
                      "BedroomAbvGr","FullBath","TotRmsAbvGrd","GarageArea","ExterQual","ExterCond","BsmtQual","BsmtCond","GarageQual","GarageCond","HeatingQC","KitchenQual"]]
y=data_select["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=800)


# -------------------------------------------standardlization-------------------------------------------
# Formula: (x-x_bar)/standard deviation
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(x_train)
xtrain_scaled = pd.DataFrame(xtrain_scaled, columns=x_train.columns)
# print(xtrain_scaled)

xtest_scaled = scaler.transform(x_test)
xtest_scaled = pd.DataFrame(xtest_scaled, columns=x_test.columns)
# print(xtest_scaled)


# -------------------------------------------Generating CSV-------------------------------------------
# data_select[["LotArea","LotFrontage","MSSubClass","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","GrLivArea",
#                       "BedroomAbvGr","FullBath","TotRmsAbvGrd","GarageArea","ExterQual","ExterCond","BsmtQual","BsmtCond","GarageQual","GarageCond","HeatingQC","KitchenQual","SalePrice"]].to_csv("data2.csv",index=False)



#  -------------------------------------------model-------------------------------------------
# Linear Regression(for testing purpose)
# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# lr.fit(xtrain_scaled,y_train)
# print(lr.score(xtrain_scaled,y_train))
# print(lr.score(xtest_scaled,y_test))

# Def ANN
def Ann(x_train, y_train, x_test, y_test):
    model = keras.Sequential([keras.layers.Dense(32, input_shape=(20,)),
                              keras.layers.Dense(64, activation=tf.nn.relu),
                              keras.layers.Dense(64, activation=tf.nn.relu),
                              keras.layers.Dense(1)])
    model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.04), loss="mean_absolute_error", metrics=["mae"])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

    return model, history

# Run ANN
model, history = Ann(xtrain_scaled, y_train, xtest_scaled, y_test)

# Learning Graph by Iterations
df_his=pd.DataFrame(history.history)
df_his[["mae","val_mae"]].plot()
plt.show()

# Accuracy Score based on R Sqaure value
ytest_pred=model.predict(xtest_scaled).flatten()
ytrain_pred=model.predict(xtrain_scaled).flatten()
print("Accuracy Score for Training: ",r2_score(ytrain_pred,y_train))
print("Accuracy Score for Testing: " ,r2_score(ytest_pred,y_test))

# Residual VS Fitted Plot
df_testresults=pd.DataFrame(np.vstack([y_test,ytest_pred])).T
df_testresults.columns=["y_test", "ytest_pred"]
df_testresults["residual"]=df_testresults["y_test"]-df_testresults["ytest_pred"]
df_testresults["Standard_Residual"]=df_testresults["residual"].transform(lambda x:(x-x.mean())/x.std())
df_testresults.head()
plt.figure(figsize=(9,6))
sns.scatterplot(x="ytest_pred",y="residual",data=df_testresults)
plt.axhline(y=0,c="red")
plt.title("Residual VS Fitted",fontsize=14)
sns.despine()
plt.show()

# ScaleLocation Plot
plt.figure(figsize=(9,6))
sns.scatterplot(x="ytest_pred",y="Standard_Residual",data=df_testresults)
plt.axhline(y=0,c="red")
plt.title("Scale-Location",fontsize=14)
sns.despine()
plt.show()


# QQ Plot
ytest_sort=sorted(df_testresults["residual"].values)
ytest_sort_p=np.arange(len(ytest_sort))/len(ytest_sort)
r=stats.norm.ppf(ytest_sort_p)
plt.figure(figsize=(9,6))
sns.scatterplot(x=r,y=sorted(df_testresults["residual"].values))
plt.title("Normal Q-Q",fontsize=14)
sns.despine()
plt.show()


#  -------------------------------------------Reference-------------------------------------------
# https://www.cnblogs.com/wj-1314/p/10461816.html
# https://www.cnblogs.com/wj-1314/p/14049195.html
# https://www.youtube.com/watch?v=8zwILUzux6o