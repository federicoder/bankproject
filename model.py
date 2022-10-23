# Support functions

from sklearn.model_selection import GridSearchCV, KFold
from preparingdata import ds_train, ds_test, ds
from utils_service import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, plot_tree
from IPython.display import Image
import matplotlib.pyplot as plt
import pydotplus
from six import StringIO
# defining x_train and y_train for this phase:
x_train, y_train = ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited

### FITTING:  tuning hyperparameters with GridSearch
### HYPERPARAMETER OPTIMIZATION
# Fit primal logistic regression
param_grid = {'C': [0.1, 0.5, 1, 10, 50, 100], 'max_iter': [250], 'fit_intercept': [True], 'intercept_scaling': [1],
              'penalty': ['l2'], 'tol': [0.00001, 0.0001, 0.000001]}
log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=10, refit=True, verbose=0)
log_primal_Grid.fit(x_train, y_train)
print("hyperparameter tuning for logistic regression:\n")
best_model(log_primal_Grid)

# Fit SVM
param_grid = {'C': [0.5, 100, 150], 'gamma': [0.1, 0.01, 0.001], 'probability': [True], 'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(x_train, y_train)
print("hyperparameter tuning for SVM:\n")
best_model(SVM_grid)

# Fit Decision Tree Classifier with max_depth=8 setted:
print("Fit Decision Tree Classifier with max depth 8 setted:\n")
param_grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}
dtree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=8), param_grid,
                          refit=True, verbose=3)
dtree_grid.fit(x_train, y_train)
print("hyperparameter tuning for Decision Tree:\n")
best_model(dtree_grid)

# Fit KNN:
param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3)
knn_grid.fit(x_train, y_train)
print("hyperparameter tuning for KNN:\n")
best_model(knn_grid)

# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6], 'max_features': [2, 4, 6]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(x_train, y_train)
print("hyperparameter tuning for Random Forest:\n")
best_model(RanFor_grid)
# Fit Extreme Gradient boosting classifier
param_grid = {'max_depth': [5, 6], 'gamma': [0.01, 0.001, 0.001], 'min_child_weight': [1, 5, 10],
              'learning_rate': [0.05, 0.1, 0.2, 0.3]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(x_train, y_train)
print("hyperparameter tuning for XGB:\n")
best_model(xgb_grid)

### FIT BEST MODELS:
# Fit primal logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                max_iter=250, n_jobs=None,
                                penalty='l2', random_state=None, solver='lbfgs', tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(x_train, y_train)

# Fit Decision Tree Classifier Standard With max_depth:8:
print("Fit Decision Tree Classifier Standard With max_depth:8:\n")
dtree = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=8)
dtree.fit(x_train, y_train)
fig = plt.figure(figsize=(160, 100))
_ = plot_tree(dtree,
              feature_names=ds_train.columns.values.tolist(),
              class_names=True,
              filled=True,
              fontsize=14,
              max_depth=12)
fig.savefig("./output/decistion_tree.png")
plt.show()

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=ds_train.drop(['Exited'], axis=1).columns,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('./output/dtree_done_with_graphiz_training_set.png')
Image(graph.create_png())
# Fit KNN:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
# Fit SVM with RBF Kernel
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1,
              kernel='rbf', max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_RBF.fit(x_train, y_train)

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=8, max_features=6,
                            min_samples_leaf=1, min_samples_split=3,
                            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0, warm_start=False)
RF.fit(x_train, y_train)

# Fit Extreme Gradient Boost Classifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0,
                    reg_lambda=1)
xgb.fit(x_train, y_train)

# cross validation section done as seen on lessons:

# CV for svc:
print("Example of CV using KFOLD applied to RandomForestClassifier (instead of using best_model function):\n ")

kfold = KFold(n_splits=5, shuffle=True, random_state=200)

randFWithCF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=8, max_features=6,
                                     min_samples_leaf=1, min_samples_split=3,
                                     min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                                     oob_score=False, random_state=None, verbose=0, warm_start=False)

randFCFWithmax_features6 = cross_val_score(randFWithCF, x_train, y_train, cv=kfold, scoring='roc_auc')

print(f'Stratified cross-validation scores with max features 6 and n_estimator 50: {randFCFWithmax_features6}')
print(f'Stratified cross-validation scores mean: {randFCFWithmax_features6.mean()}')


print("Classification Reports of all models in training phase:\n")
print(classification_report(ds_train.Exited, log_primal.predict(x_train)))
print(classification_report(ds_train.Exited, SVM_RBF.predict(x_train)))
print(classification_report(ds_train.Exited, dtree.predict(x_train)))
print(classification_report(ds_train.Exited, knn.predict(x_train)))
print(classification_report(ds_train.Exited, RF.predict(x_train)))
print(classification_report(ds_train.Exited, xgb.predict(x_train)))
