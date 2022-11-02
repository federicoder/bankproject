import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter
import utils_service as us
from imblearn.under_sampling import NearMiss
import seaborn as sns
import sys
sns.set(palette="Set2")
matplotlib.use('TkAgg')

ds = pd.read_csv("./dataset/Churn_Modelling.csv")
# casting to int64 float attributes:
ds['EstimatedSalary'] = ds['EstimatedSalary'].astype(np.int64)
ds['Balance'] = ds['Balance'].astype(np.int64)
# remove unrelevant features:
ds.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
# first five row of the dataset
print(ds.head())
print(ds.dtypes)
# to see how labels:
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
# so now i take the not churnered people:
new_ds = ds[ds['Exited'] == 0]
new_ds['BalanceSalaryRatio'] = new_ds.Balance / new_ds.EstimatedSalary

# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
new_ds['TenureByAge'] = new_ds.Tenure / (new_ds.Age)
new_ds['CreditScoreGivenAge'] = new_ds.CreditScore / (new_ds.Age)
# new_ds['EstimatedSalary'] = us.remove_outliers(new_ds, 'EstimatedSalary')

# Arrange columns by data type for easier manipulation
continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']
new_ds = new_ds[['Exited'] + continuous_vars + cat_vars]
new_ds.head()

'''For the one hot variables, we change 0 to -1 so that the models can capture a negative relation 
where the attribute in inapplicable instead of 0'''
new_ds.loc[new_ds.HasCrCard == 0, 'HasCrCard'] = -1
new_ds.loc[new_ds.IsActiveMember == 0, 'IsActiveMember'] = -1
new_ds.head()

# Converting values of geography and gender from string to float (to fit the model next time)
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (new_ds[i].dtype == np.str or new_ds[i].dtype == np.object):
        for j in new_ds[i].unique():
            new_ds[i + '_' + j] = np.where(new_ds[i] == j, 1, -1)
        remove.append(i)
new_ds = new_ds.drop(remove, axis=1)
new_ds.head()

# min - max normalization:
# minMax scaling the continuous variables
print("Min-Max normalization:\n")
minVec = new_ds[continuous_vars].min().copy()
maxVec = new_ds[continuous_vars].max().copy()
new_ds[continuous_vars] = (new_ds[continuous_vars] - minVec) / (maxVec - minVec)
print(new_ds)
