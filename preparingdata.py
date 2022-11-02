import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split

pd.options.display.max_rows = None
pd.options.display.max_columns = None
import tkinter

sys.stdout = open('./output/report.txt', 'wt')
sns.set(palette="Set2")
matplotlib.use('TkAgg')

## Read the data frame
ds = pd.read_csv("./dataset/Churn_Modelling.csv")

### DATA ANALISYS

## Show number of rows and columns
print('number of rows, number of columns ')
print(ds.shape)
print('--------------------------------------------')

## Check column list and missing values
print('Check column list and missing values:')
print(ds.isnull().sum())
print('--------------------------------------------')

## Get unique count for each variable
print('Get unique count for each variable:')
print(ds.nunique())
print('--------------------------------------------')

## Get duplicated data
print('Get duplicated data:')
print(ds[ds.duplicated()])
print('--------------------------------------------')

## Check variable data types
print('Check variable data types:')
print(ds.dtypes)
print('--------------------------------------------')

## Show first five rows of the dataset
print('First five rows of the dataset:')
print(ds.head())
print('--------------------------------------------')

## Proportion of customer churned and retained
labels = 'Exited', 'Retained'
sizes = [ds.Exited[ds['Exited'] == 1].count(), ds.Exited[ds['Exited'] == 0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size=20)
plt.savefig("./output/proportions_of_customer_churned.png")
plt.show()

## Relations based on the categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue='Exited', data=ds, ax=axarr[0][0])
sns.countplot(x='Gender', hue='Exited', data=ds, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue='Exited', data=ds, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue='Exited', data=ds, ax=axarr[1][1])
plt.savefig("./output/proportions_RelevantCategoricalFeatures_Exited.png")

## Relations based on the continuous variables
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore', x='Exited', hue='Exited', data=ds, ax=axarr[0][0])
sns.boxplot(y='Age', x='Exited', hue='Exited', data=ds, ax=axarr[0][1])
sns.boxplot(y='Tenure', x='Exited', hue='Exited', data=ds, ax=axarr[1][0])
sns.boxplot(y='Balance', x='Exited', hue='Exited', data=ds, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts', x='Exited', hue='Exited', data=ds, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary', x='Exited', hue='Exited', data=ds, ax=axarr[2][1])
plt.savefig("./output/proportions_RelevantContinuousFeatures_Exited.png")

## Checking correlation:
plt.subplots(figsize=(11, 8))
sns.heatmap(ds.corr(), annot=True, cmap="RdYlBu")
plt.savefig("./output/correlation.png")
plt.show()

### PRE-PROCESSING

## Remove unrelevant features:
ds.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

## Casting to int64 float attributes:
ds['EstimatedSalary'] = ds['EstimatedSalary'].astype(np.int64)
ds['Balance'] = ds['Balance'].astype(np.int64)

## Split Dataset in Train set and Test set:
ds_train = ds.sample(frac=0.8, random_state=42)
ds_test = ds.drop(ds_train.index)
print('Total number of rows for the Training set and the Test set:')
print(len(ds_train))
print(len(ds_test))
print('--------------------------------------------')

## Add new feature:
# Balance Salary Ratio:
ds_train['BalanceSalaryRatio'] = ds_train.Balance / ds_train.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio', x='Exited', hue='Exited', data=ds_train)
plt.ylim(-1, 5)
# Tenure By Age:
ds_train['TenureByAge'] = ds_train.Tenure / (ds_train.Age)
sns.boxplot(y='TenureByAge', x='Exited', hue='Exited', data=ds_train)
plt.ylim(-1, 1)
plt.savefig("./output/ds_with_new_feature_TenureByAge.png")
plt.show()
# Credit Score Given Age:
ds_train['CreditScoreGivenAge'] = ds_train.CreditScore / (ds_train.Age)
print('First five rows of the Training Set with the new Features:')
print(ds_train.head())
print('--------------------------------------------')

## Detecting outliers:
print('detecting outliers:\n')
plt.figure(figsize=(12, 12))
bplot = ds.boxplot(patch_artist=True)
plt.xticks(rotation=90)
plt.savefig("./output/outliers.png")
plt.show()

## Reorder the columns
continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']
ds_train = ds_train[['Exited'] + continuous_vars + cat_vars]
print('First five rows of the Reordered Training Set:')
print(ds_train.head())
print('--------------------------------------------')

## Change the 0 in categorical variables to -1
ds_train.loc[ds_train.HasCrCard == 0, 'HasCrCard'] = -1
ds_train.loc[ds_train.IsActiveMember == 0, 'IsActiveMember'] = -1
print('First five rows of the Training Set after changing the HasCrCard and IsActiveMember values from 0 to -1:')
print(ds_train.head())
print('--------------------------------------------')

## One hot encode the categorical variables
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (ds_train[i].dtype == np.str or ds_train[i].dtype == np.object):
        for j in ds_train[i].unique():
            ds_train[i + '_' + j] = np.where(ds_train[i] == j, 1, -1)
        remove.append(i)
ds_train = ds_train.drop(remove, axis=1)
print('First five rows of the Training Set after One Hot Encoding:')
print(ds_train.head())
print('--------------------------------------------')

## Min-Max Normalization (min-max scaling the continuous variables):
print("Min-Max normalization:\n")
minVec = ds_train[continuous_vars].min().copy()
maxVec = ds_train[continuous_vars].max().copy()
ds_train[continuous_vars] = (ds_train[continuous_vars] - minVec) / (maxVec - minVec)
print('First five rows of the Training Set after min-max normalization:')
print(ds_train.head())
print('--------------------------------------------')

## Preparing Training set for GMM:
ds_gmm = ds.copy()

## Add new feature:
# Balance Salary Ratio:
ds_gmm['BalanceSalaryRatio'] = ds_gmm.Balance / ds_gmm.EstimatedSalary

ds_gmm['TenureByAge'] = ds_gmm.Tenure / (ds_gmm.Age)
# Credit Score Given Age:
ds_gmm['CreditScoreGivenAge'] = ds_gmm.CreditScore / (ds_gmm.Age)

## Reorder the columns
continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']
ds_gmm = ds_gmm[['Exited'] + continuous_vars + cat_vars]

## Change the 0 in categorical variables to -1
ds_gmm.loc[ds_gmm.HasCrCard == 0, 'HasCrCard'] = -1
ds_gmm.loc[ds_gmm.IsActiveMember == 0, 'IsActiveMember'] = -1

## One hot encode the categorical variables
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (ds_gmm[i].dtype == np.str or ds_gmm[i].dtype == np.object):
        for j in ds_gmm[i].unique():
            ds_gmm[i + '_' + j] = np.where(ds_gmm[i] == j, 1, -1)
        remove.append(i)
ds_gmm = ds_gmm.drop(remove, axis=1)

## Min-Max Normalization (min-max scaling the continuous variables):
minVec2 = ds_gmm[continuous_vars].min().copy()
maxVec2 = ds_gmm[continuous_vars].max().copy()
ds_gmm[continuous_vars] = (ds_gmm[continuous_vars] - minVec2) / (maxVec2 - minVec2)

df_0 = ds_gmm[ds_gmm['Exited']== 0]
df_1 = ds_gmm[ds_gmm['Exited']== 1]

# Split non-exited data in 90% for training GMM and 10% for cross-validation and testing Hold OUT!
X_N_train, X_N_cv_test, y_N_train, y_N_cv_test = train_test_split(df_0.drop(['Exited'],axis=1), df_0['Exited'], test_size=0.1, random_state=1)
# Split the exited data in 50% for cross-validation and 50% for testing
X_F_cv, X_F_test, y_F_cv, y_F_test = train_test_split(df_1.drop(['Exited'],axis=1), df_1['Exited'], test_size=0.5, random_state=1)
# Split the remaining 10% non-exited in 50% for cross-validation and 50% for testing
X_N_cv, X_N_test, y_N_cv, y_N_test = train_test_split(X_N_cv_test, y_N_cv_test , test_size=0.5, random_state=1)

# Generate the 3 new datasets (Train + CV + test)
X_CV = np.vstack([X_N_cv, X_F_cv])
y_CV = np.hstack([y_N_cv, y_F_cv])
X_test = np.vstack([X_N_test, X_F_test])
y_test = np.hstack([y_N_test, y_F_test])


# siccome i dati erano 10000 record in ds e gli exited sono 2000 (20%), il 10% per il test alla fine sarà una percetuale minima


