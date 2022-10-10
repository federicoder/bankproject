# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from preparingdata import ds_train, ds_test, ds
from utils_service import *
# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils_service import compute_and_print_evaluation

### FITTING:  tuning hyperparameters with GridSearch
### HYPERPARAMETER OPTIMIZATION
# Fit primal logistic regression
param_grid = {'C': [0.1, 0.5, 1, 10, 50, 100], 'max_iter': [250], 'fit_intercept': [True], 'intercept_scaling': [1],
              'penalty': ['l2'], 'tol': [0.00001, 0.0001, 0.000001]}
log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=10, refit=True, verbose=0)
log_primal_Grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(log_primal_Grid)

# Fit SVM
param_grid = {'C': [0.5, 100, 150], 'gamma': [0.1, 0.01, 0.001], 'probability': [True], 'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(SVM_grid)

# Fit Decision Tree Classifier:
param_grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}
dtree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit=True, verbose=3)
dtree_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(dtree_grid)

# Fit KNN:
param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3)
knn_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(knn_grid)

# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6], 'max_features': [2, 4, 6]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(RanFor_grid)
# Fit Extreme Gradient boosting classifier
param_grid = {'max_depth': [5, 6], 'gamma': [0.01, 0.001, 0.001], 'min_child_weight': [1, 5, 10],
              'learning_rate': [0.05, 0.1, 0.2, 0.3]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(xgb_grid)

### FIT BEST MODELS:
# Fit primal logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                                max_iter=250, n_jobs=None,
                                penalty='l2', random_state=None, solver='lbfgs', tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)

# Fit Decision Tree Classifier:
dtree = DecisionTreeClassifier()
dtree.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
# Fit KNN:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
# Fit SVM with RBF Kernel
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1,
              kernel='rbf', max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_RBF.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=8, max_features=6,
                            min_samples_leaf=1, min_samples_split=3,
                            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0, warm_start=False)
RF.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
# Fit Extreme Gradient Boost Classifier
xgb = XGBClassifier()
xgb.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
# xgb_pred = xgb.predict(ds_test.loc[:, ds_test.columns != 'Exited'])


print(classification_report(ds_train.Exited, log_primal.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited, SVM_RBF.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited, dtree.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited, knn.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
# # print(classification_report(ds_train.Exited, SVM_POL.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited, RF.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited, xgb.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
