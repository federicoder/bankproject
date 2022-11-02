# Support functions
from preparingdata import minVec, maxVec, ds_test, X_test, y_test
from model import *
import matplotlib.pyplot as plt
import matplotlib
from utils_service import *

matplotlib.use("TkAgg")

# # get auc scores for training set:
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
plt.plot(fpr_KNN, tpr_KNN, label='KNN RBF Score: ' + str(round(auc_KNN, 5)))
plt.plot(fpr_DTree, tpr_Dtree, label='Decision Tree RBF Score: ' + str(round(auc_DTree, 5)))
plt.plot(fpr_RF, tpr_RF, label='RF score: ' + str(round(auc_RF, 5)))
plt.plot(fpr_XGB, tpr_XGB, label='XGB score: ' + str(round(auc_XGB, 5)))
plt.plot([0, 1], [0, 1], 'k--', label='Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig("./output/roc_results_ratios_on_train_set.png")
plt.show()

# get auc scores for test set:
# TESTING PHASE:
# Make the data transformation for test data
print('Test Phase:\n')
ds_test = df_prep_pipeline(ds_test, ds_train.columns, minVec, maxVec)
ds_test = ds_test.mask(np.isinf(ds_test))
ds_test = ds_test.dropna()
print(ds_test.shape)


# getting some information about classification pararmeters for evaluation phase:
print("Evaluation parameters for Random Forest:\n")
compute_and_print_evaluation(ds_test.Exited, RF.predict(ds_test.loc[:, ds_test.columns != 'Exited']))
print('--------------------------------------------')
print("Evaluation parameters for XGB:\n")
compute_and_print_evaluation(ds_test.Exited, xgb.predict(ds_test.loc[:, ds_test.columns != 'Exited']))
print('--------------------------------------------')
print("Evaluation parameters for Dtree:\n")
compute_and_print_evaluation(ds_test.Exited, dtree.predict(ds_test.loc[:, ds_test.columns != 'Exited']))
print('--------------------------------------------')

auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(ds_test.Exited,
                                                       RF.predict(ds_test.loc[:, ds_test.columns != 'Exited']),
                                                       RF.predict_proba(ds_test.loc[:, ds_test.columns != 'Exited'])[:,
                                                       1])
auc_DTree_test, fpr_DTree_test, tpr_Dtree_test = get_auc_scores(ds_test.Exited,
                                                                dtree.predict(
                                                                    ds_test.loc[:, ds_test.columns != 'Exited']),
                                                                dtree.predict_proba(
                                                                    ds_test.loc[:, ds_test.columns != 'Exited'])[:,
                                                                1])

auc_xgb_test, fpr_xgb_test, tpr_xgb_test = get_auc_scores(ds_test.Exited,
                                                          xgb.predict(
                                                              ds_test.loc[:, ds_test.columns != 'Exited']),
                                                          xgb.predict_proba(
                                                              ds_test.loc[:, ds_test.columns != 'Exited'])[:,
                                                          1])

plt.figure(figsize=(12, 6), linewidth=1)
plt.plot(fpr_RF_test, tpr_RF_test, label='RF score: ' + str(round(auc_RF_test, 5)))
plt.plot(fpr_DTree_test, tpr_Dtree_test, label='Dtree score: ' + str(round(auc_DTree_test, 5)))
plt.plot(fpr_xgb_test, tpr_xgb_test, label='XGB score: ' + str(round(auc_xgb_test, 5)))

plt.plot([0, 1], [0, 1], 'k--', label='Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig("./output/roc_results_ratios_on_test_set.png")
plt.show()

# Getting information about GMM threshold:
# Get the score for each sample
print('Getting infos for GMM thresholds and score in test phase:\n')
plt.plot(T_vec, aucpr_vs_t)
plt.plot(T_vec, precision_vs_t)
plt.plot(T_vec, recall_vs_t)
ax = plt.gca()
ax.set(title='Evolution of performance scores vs. threshold for GMM probability',
       xlabel='Threshold T [neg loglikelihood]')
ax.legend(['AUCPR (5 fold CV average)', 'Precision (5 fold CV average)', 'Recall (5 fold CV average)'])
ax.invert_xaxis()
plt.savefig("./output/evaluation_for_gmm.png")
plt.show()
print('Maximum cross validation AUCPR=' + str(max(aucpr_vs_t)))
T_opt = T_vec[np.argmax(aucpr_vs_t)]
print('Optimal threshold T = ' + str(T_opt))

y_test_proba = gmm.score_samples(X_test)
y_test_pred = y_test_proba.copy()
y_test_pred[y_test_pred >= T_opt] = 0
y_test_pred[y_test_pred < T_opt] = 1

test_precision = (precision_score(y_test, y_test_pred))
test_recall = (recall_score(y_test, y_test_pred))
test_aucpr = (average_precision_score(y_test, y_test_pred))
print("TEST results --> aucpr=%.3f - Precision=%.3f - Recall=%.3f" % (test_aucpr, test_precision, test_recall))
