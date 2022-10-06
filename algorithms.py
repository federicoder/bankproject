import matplotlib.pyplot as plt

from preparingdata import ds
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score, )
from mlxtend.plotting import plot_confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
import pandas as pd

# Prediction with ML models:
X = ds.drop("Exited", axis=1)
y = ds["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# for gaussian:
clf = GaussianNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
# for logistic reg:
clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
# for decision tree:
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
# for decision tree:
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
# for random forest:
clf = RandomForestClassifier(n_estimators=200, random_state=200)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
# for Extreme gradient booster:
clf = XGBClassifier(max_depth=10, random_state=10, n_estimators=220, eval_metric='auc', min_child_weight=3,
                    colsample_bytree=0.75, subsample=0.9)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)

# Scaling:
scaler = MinMaxScaler()

bumpy_features = ["CreditScore", "Age", "Balance", 'EstimatedSalary']

df_scaled = pd.DataFrame(data=X)
df_scaled[bumpy_features] = scaler.fit_transform(X[bumpy_features])

# over sampling:
X = df_scaled
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=7)
clf = XGBClassifier(max_depth=12, random_state=7, n_estimators=100, eval_metric='auc', min_child_weight=3,
                    colsample_bytree=0.75, subsample=0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Parameters for XBG:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))

clf = tree.DecisionTreeClassifier(max_depth=12, random_state=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mytree = tree.plot_tree(decision_tree=clf, max_depth=5, class_names="ciccio", fontsize=12)
tree.export_graphviz(decision_tree=clf, max_depth=5, out_file='./output/myfile.png')
print("\nParameters for Decision Tree:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))


# plt.plot()
# plt.savefig('./output/decisiontreeParameters.png')
# Confusion Matrix
confusion_matrix(y_test, y_pred)
