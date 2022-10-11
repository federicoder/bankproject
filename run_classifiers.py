# Support functions
from preparingdata import minVec, maxVec
from model import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report

matplotlib.use("TkAgg")
y = ds_train.Exited
X = ds_train.loc[:, ds_train.columns != 'Exited']
auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),
                                                                log_primal.predict_proba(X)[:, 1])
auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(y, SVM_RBF.predict(X), SVM_RBF.predict_proba(X)[:, 1])
auc_KNN, fpr_KNN, tpr_KNN = get_auc_scores(y, knn.predict(X), knn.predict_proba(X)[:, 1])
auc_DTree, fpr_DTree, tpr_Dtree = get_auc_scores(y, dtree.predict(X), dtree.predict_proba(X)[:, 1])
auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X), RF.predict_proba(X)[:, 1])
auc_XGB, fpr_XGB, tpr_XGB = get_auc_scores(y, xgb.predict(X), xgb.predict_proba(X)[:, 1])

plt.figure(figsize=(12, 6), linewidth=1)
plt.plot(fpr_log_primal, tpr_log_primal, label='log primal Score: ' + str(round(auc_log_primal, 5)))
plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label='SVM RBF Score: ' + str(round(auc_SVM_RBF, 5)))
plt.plot(fpr_KNN, tpr_KNN, label='KNN RBF Score: ' + str(round(auc_SVM_RBF, 5)))
plt.plot(fpr_DTree, tpr_Dtree, label='Decision Tree RBF Score: ' + str(round(auc_SVM_RBF, 5)))
plt.plot(fpr_RF, tpr_RF, label='RF score: ' + str(round(auc_RF, 5)))
plt.plot(fpr_XGB, tpr_XGB, label='XGB score: ' + str(round(auc_XGB, 5)))
plt.plot([0, 1], [0, 1], 'k--', label='Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig("./output/roc_results_ratios_on_train_set.png")
plt.show()

# TESTING PHASE:
# Make the data transformation for test data
ds_test = df_prep_pipeline(ds_test, ds_train.columns, minVec, maxVec)
ds_test = ds_test.mask(np.isinf(ds_test))
ds_test = ds_test.dropna()
print(ds_test.shape)
print(classification_report(ds_test.Exited, RF.predict(ds_test.loc[:, ds_test.columns != 'Exited'])))
auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(ds_test.Exited,
                                                       RF.predict(ds_test.loc[:, ds_test.columns != 'Exited']),
                                                       RF.predict_proba(ds_test.loc[:, ds_test.columns != 'Exited'])[:,
                                                       1])
plt.figure(figsize=(12, 6), linewidth=1)
plt.plot(fpr_RF_test, tpr_RF_test, label='RF score: ' + str(round(auc_RF_test, 5)))
plt.plot([0, 1], [0, 1], 'k--', label='Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig("./output/roc_results_ratios_on_test_set.png")
plt.show()
