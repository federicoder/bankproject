import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils_service import *
from preparing_data_for_gmm import *
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from preparingdata import *
from sklearn.datasets._samples_generator import make_blobs

x_train, y_train = new_ds.loc[:, new_ds.columns != 'Exited'], new_ds.Exited

# # computing aic and bic to define the number of cluster:
# # Create empty lists to store the BIC and AIC values
# bic_score = []
# aic_score = []
# # Set up a range of cluster numbers to try
# n_range = range(2,11)
# # Loop through the range and fit a model
# for n in n_range:
#     gm = GaussianMixture(n_components=n,
#                          random_state=123,
#                          n_init=10)
#     gm.fit(x_train)
#
#     # Append the BIC and AIC to the respective lists
#     bic_score.append(gm.bic(x_train))
#     aic_score.append(gm.aic(x_train))
#
# # Plot the BIC and AIC values together
# fig, ax = plt.subplots(figsize=(12, 8), nrows=1)
# ax.plot(n_range, bic_score, '-o', color='orange')
# ax.plot(n_range, aic_score, '-o', color='green')
# ax.set(xlabel='Number of Clusters', ylabel='Score')
# ax.set_xticks(n_range)
# ax.set_title('BIC and AIC Scores Per Number Of Clusters')
# plt.show()
#

tuned_parameters = {'n_components': np.array([1, 2, 3, 4])}
# gmm_grid_without_CV = GaussianMixture()
#
# gmm_grid_without_CV.fit(x_train, None)
# gmm_grid_without_CV.predict(x_train)
# compute_and_print_evaluation(y_train, gmm_grid_without_CV.predict(x_train))
gmm_grid = GridSearchCV(GaussianMixture(), param_grid=tuned_parameters, cv=6, refit=True, verbose=3)
gmm_grid.fit(x_train, None)
plt.scatter(gmm_grid.cv_results_['param_n_components'], \
            gmm_grid.cv_results_['rank_test_score'])
plt.show()
print("hyperparameter tuning for Gaussian Mixture Model:\n")
best_model(gmm_grid)

# new_x_train, new_y_train = ds_train.loc[:, ds_train.columns != 'Exited'], ds_train.Exited
gmm = GaussianMixture(verbose=True, random_state=200)
gmm.fit(x_train.values, None)
labels = gmm.predict(x_train)
# compute_and_print_evaluation(x_train, labels)

# Test
ds_test = df_prep_pipeline(ds_test, ds_train.columns, minVec, maxVec)
ds_test = ds_test.mask(np.isinf(ds_test))
ds_test = ds_test.dropna()
print(ds_test.shape)
sizes = [ds_test.Exited[ds_test['Exited'] == 1].count(), ds_test.Exited[ds_test['Exited'] == 0].count()]
print('sizes of ds_Test:\n')
print(sizes)

# Get the score for each sample
score = gmm.score_samples(ds_test.loc[:, ds_test.columns != 'Exited'])
print(score)
# Save score as a column
ds_test['score'] = score
# Get the score threshold for anomaly
pct_threshold = np.percentile(score, 19.48)
# Print the score threshold
print(f'The threshold of the score is {pct_threshold:.2f}')
# Label the anomalies
ds_test['anomaly_gmm_pct'] = ds_test['score'].apply(lambda x: 1 if x < pct_threshold else 0)
print(ds_test['anomaly_gmm_pct'].values)
print('--------------------------------------------')

# Visualize the actual and predicted anomalies
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(20, 12))
# Ground truth
ax0.set_title('Ground Truth')
ax0.scatter(ds_test['BalanceSalaryRatio'], ds_test['NumOfProducts'], c=ds_test['Exited'], cmap='rainbow')
# GMM Predictions
ax1.set_title('GMM Predict Anomalies Using Percentage')
ax1.scatter(ds_test['BalanceSalaryRatio'], ds_test['NumOfProducts'], c=ds_test['anomaly_gmm_pct'], cmap='rainbow')
plt.show()
plt.savefig('./output/gmm_precidciton_with_test_set.png')