from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
import optuna
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import ElasticNet

xgb.config_context(verbosity= 1) # silent
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_elasticNet(trial):
    """
    Optuna objective function for hyperparameter tuning of a regression model.

    Args:
        trial: An Optuna `Trial` object used to sample hyperparameters.
        x_train: A numpy array of shape `(n_samples, n_features)` containing
            the training data.
        y_train: A numpy array of shape `(n_samples,)` containing the target
            values for the training data.
        x_test: A numpy array of shape `(n_samples, n_features)` containing
            the test data.
        y_test: A numpy array of shape `(n_samples,)` containing the target
            values for the test data.

    Returns:
        The root mean squared error (RMSE) of the regression model on the test
            data.
    """
    # Import data
    df_train = pd.read_csv('/exports/reum/tdmaarseveen/FRYQ_vragenlijst/proc/df_train.csv', sep=';')
    
    # Define labels/ input
    target = 'label' # _1y
    cols_data = [x for x in list(df_train.columns) if x not in ['Category', 'ZDNnummer', 'VERHA', target]] #   'Sex', 'Age', 
    
    X = df_train[cols_data]
    y = df_train[target].replace({0: False, 1: True})
    
    # Bookmark all predictions
    y_pred = []
    
    # Perform kfold CV
    kf = KFold(n_splits=5) 
    
    for train_index, test_index in kf.split(X):
        train_x, test_x = pd.DataFrame(X).loc[train_index], pd.DataFrame(X).loc[test_index]
        train_y, test_y = np.take(y, np.array(train_index)), np.take(y, np.array(test_index))

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 100, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0, step=0.01),
            # Add tolerance ?
        }
        model = ElasticNet(**params)
        model.fit(train_x, train_y)
        #mlflow.sklearn.log_model(model, "model")
        y_pred.extend(model.predict(test_x))
        
    # Check performance
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    #ppv = precision_score(y, y_pred)
    print(trial._trial_id)
    return rmse

#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=100) # test # 2000 # -1 -> bootstrapping (take random 100 samples) , n_jobs=1

study = optuna.create_study(direction="minimize")
study.optimize(objective_elasticNet, n_trials=1000) # test # 500 # -1 -> bootstrapping (take random 100 samples) , n_jobs=1

print('Best trial: %s' % study.best_trial.number)
print('Performance (accuracy): %s' % study.best_trial.value)
print('Corresponding Parameters: %s' % study.best_trial.params)


import plotly as py
# optuna.visualization.plot_intermediate_values(study)
fig = optuna.visualization.plot_optimization_history(study)
py.offline.plot(fig, filename='hyperparamtuning_optimization_elasticNet_new.html', auto_open=False)

fig = optuna.visualization.plot_contour(study)
py.offline.plot(fig, filename='hyperparamtuning_contour_elasticNet_new.html', auto_open=False)