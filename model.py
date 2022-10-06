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

### FITTING:
# Fit primal logistic regression
param_grid = {'C': [0.1, 0.5, 1, 10, 50, 100], 'max_iter': [250], 'fit_intercept': [True], 'intercept_scaling': [1],
              'penalty': ['l2'], 'tol': [0.00001, 0.0001, 0.000001]}
log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=10, refit=True, verbose=0)
log_primal_Grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited)
best_model(log_primal_Grid)
# Fit logistic regression with degree 2 polynomial kernel
param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
              'tol':[0.0001,0.000001]}
poly2 = PolynomialFeatures(degree=2)
ds_train_pol2 = poly2.fit_transform(ds_train.loc[:, ds_train.columns != 'Exited'])
log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
log_pol2_Grid.fit(ds_train_pol2,ds_train.Exited)
best_model(log_pol2_Grid)
# Fit SVM with RBF Kernel
param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
best_model(SVM_grid)
# Fit SVM with pol kernel
param_grid = {'C': [0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['poly'],'degree':[2,3] }
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
best_model(SVM_grid)
# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
best_model(RanFor_grid)
# Fit Extreme Gradient boosting classifier
param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
best_model(xgb_grid)
### FIT BEST MODELS:
# Fit primal logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='warn',n_jobs=None,
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
# Fit logistic regression with pol 2 kernel
poly2 = PolynomialFeatures(degree=2)
ds_train_pol2 = poly2.fit_transform(ds_train.loc[:, ds_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='warn', n_jobs=None,
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(ds_train_pol2,ds_train.Exited)
# Fit SVM with RBF Kernel
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True,
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
# Fit SVM with Pol Kernel
SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',  max_iter=-1,
              probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_POL.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
# Fit Random Forest classifier
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
# Fit Extreme Gradient Boost Classifier
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, missing=None, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
XGB.fit(ds_train.loc[:, ds_train.columns != 'Exited'],ds_train.Exited)
print(classification_report(ds_train.Exited, log_primal.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited,  log_pol2.predict(ds_train_pol2)))
print(classification_report(ds_train.Exited,  SVM_RBF.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited,  SVM_POL.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited,  RF.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))
print(classification_report(ds_train.Exited,  XGB.predict(ds_train.loc[:, ds_train.columns != 'Exited'])))

