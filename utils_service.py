from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score, roc_curve, precision_score
import numpy as np


def remove_outliers(df, col):
    lower_quantile = df[col].quantile(0.25)
    upper_quantile = df[col].quantile(0.75)
    IQR = upper_quantile - lower_quantile
    lower_whisker = lower_quantile - 1.5 * IQR
    upper_whisker = upper_quantile + 1.5 * IQR
    temp = df.loc[(df[col] > lower_whisker) & (df[col] < upper_whisker)]
    return temp[col]


def compute_and_print_evaluation(y_test, predictor):
    print("ROC AUC Score:", roc_auc_score(y_test, predictor))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictor))
    print("Accuracy Score:", accuracy_score(y_test, predictor))
    print("Precision Score:", precision_score(y_test, predictor))
    print("Classification Report:\n", classification_report(y_test, predictor))


# Function to give best model score and parameters
def best_model(model):
    print("Best score of model:")
    print(model.best_score_, "\n")
    print("Best params of model:")
    print(model.best_params_, "\n")
    print("Best estimator of model:")
    print(model.best_estimator_, "\n")


def get_auc_scores(y_actual, method, method2):
    auc_score = roc_auc_score(y_actual, method)
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2)
    return auc_score, fpr_df, tpr_df


# data prep pipeline for test data
def df_prep_pipeline(ds_predict, ds_train_Cols, minVec, maxVec):
    # Add new features
    ds_predict['BalanceSalaryRatio'] = ds_predict.Balance / ds_predict.EstimatedSalary
    ds_predict['TenureByAge'] = ds_predict.Tenure / (ds_predict.Age - 18)
    ds_predict['CreditScoreGivenAge'] = ds_predict.CreditScore / (ds_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                       'BalanceSalaryRatio',
                       'TenureByAge', 'CreditScoreGivenAge']
    cat_vars = ['HasCrCard', 'IsActiveMember', "Geography", "Gender"]
    ds_predict = ds_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    ds_predict.loc[ds_predict.HasCrCard == 0, 'HasCrCard'] = -1
    ds_predict.loc[ds_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in ds_predict[i].unique():
            ds_predict[i + '_' + j] = np.where(ds_predict[i] == j, 1, -1)
        remove.append(i)
    ds_predict = ds_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(ds_train_Cols) - set(ds_predict.columns))
    for l in L:
        ds_predict[str(l)] = -1
    # MinMax scaling coontinuous variables based on min and max from the train data
    ds_predict[continuous_vars] = (ds_predict[continuous_vars] - minVec) / (maxVec - minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    ds_predict = ds_predict[ds_train_Cols]
    return ds_predict
