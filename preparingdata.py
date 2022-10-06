import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter
import utils_service as us
from imblearn.under_sampling import NearMiss
import seaborn as sns

sns.set(palette="Set2")
matplotlib.use('TkAgg')

ds = pd.read_csv("./dataset//Churn_Modelling.csv")

### DATA ANALISYS
print('number of rows, number of columns ')
print(ds.shape)
# checking column list and missing values:
print('checking column list and missing values:')
print(ds.isnull().sum())
print('Get unique count for each variable:')
print(ds.nunique())
print('Get duplicated data:')
print(ds[ds.duplicated()])
print(ds.corr())

#casting to int64 float attributes:
ds['EstimatedSalary'] = ds['EstimatedSalary'].astype(np.int64)
ds['Balance'] = ds['Balance'].astype(np.int64)

# remove unrelevant features:
ds.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
# first five row of the dataset
print(ds.head())
print(ds.dtypes)

# Here our main interest is to get an understanding as to how the given attributes relate too the 'Exit' status.

labels = 'Exited', 'Retained'
sizes = [ds.Exited[ds['Exited'] == 1].count(), ds.Exited[ds['Exited'] == 0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size=20)
plt.show()

# We first review the 'Status' relation with categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue='Exited', data=ds, ax=axarr[0][0])
sns.countplot(x='Gender', hue='Exited', data=ds, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue='Exited', data=ds, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue='Exited', data=ds, ax=axarr[1][1])

### FEATURE ENGINEERING :

# detecting outliers:
print('detecting outliers:\n')
plt.figure(figsize=(12, 12))
bplot = ds.boxplot(patch_artist=True)
plt.xticks(rotation=90)
plt.show()
plt.savefig("./output/outliers.png")

# r removing outlier from estimatedSalary:
# Split Train, test data:
ds_train = ds.sample(frac=0.8, random_state=200)
ds_test = ds.drop(ds_train.index)
print(len(ds_train))
print(len(ds_test))

# add new feature more relevant:
ds_train['BalanceSalaryRatio'] = ds_train.Balance / ds_train.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio', x='Exited', hue='Exited', data=ds_train)
plt.ylim(-1, 5)

# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
ds_train['TenureByAge'] = ds_train.Tenure / (ds_train.Age)
sns.boxplot(y='TenureByAge', x='Exited', hue='Exited', data=ds_train)
plt.ylim(-1, 1)
plt.show()

# Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
ds_train['CreditScoreGivenAge'] = ds_train.CreditScore /(ds_train.Age)

print(ds_train)

# removing outliers:
ds_train['EstimatedSalary'] = us.remove_outliers(ds, 'EstimatedSalary')
sns.boxenplot(ds['EstimatedSalary'])
plt.ylabel('distribution')
plt.show()

# checking correlation:
plt.subplots(figsize=(11, 8))
sns.heatmap(ds.corr(), annot=True, cmap="RdYlBu")
plt.show()
plt.savefig("./output/correlation.png")

# Arrange columns by data type for easier manipulation
continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']
ds_train = ds_train[['Exited'] + continuous_vars + cat_vars]
ds_train.head()

'''For the one hot variables, we change 0 to -1 so that the models can capture a negative relation 
where the attribute in inapplicable instead of 0'''
ds_train.loc[ds_train.HasCrCard == 0, 'HasCrCard'] = -1
ds_train.loc[ds_train.IsActiveMember == 0, 'IsActiveMember'] = -1
ds_train.head()

# Converting values of geography and gender from string to float (to fit the model next time)
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (ds_train[i].dtype == np.str or ds_train[i].dtype == np.object):
        for j in ds_train[i].unique():
            ds_train[i + '_' + j] = np.where(ds_train[i] == j, 1, -1)
        remove.append(i)
ds_train = ds_train.drop(remove, axis=1)
ds_train.head()


# min - max normalization:
# minMax scaling the continuous variables
minVec = ds_train[continuous_vars].min().copy()
maxVec = ds_train[continuous_vars].max().copy()
ds_train[continuous_vars] = (ds_train[continuous_vars]-minVec)/(maxVec-minVec)
ds_train.head()