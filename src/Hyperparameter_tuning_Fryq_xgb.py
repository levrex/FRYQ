from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
import optuna
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score, balanced_accuracy_score, roc_auc_score

from sklearn.linear_model import ElasticNet


xgb.config_context(verbosity= 1) # silent
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_xgb(trial):
    
    # Import data
    df_train = pd.read_csv('/exports/reum/tdmaarseveen/FRYQ_vragenlijst/proc/df_train.csv', sep=';')
    
    # add preprocessing? 
    
    # Define labels/ input
    target = 'label' # _1y
    
    # All data?
    #cols_data = [x for x in list(df_train.columns) if x not in ['Category', 'ZDNnummer', 'VERHA', target]] #   'Sex', 'Age', 
    
    # elastic net?
    cols_data = ['v1.4', 'v1.7', 'v1.9', 'v1.10', 'v1.11', 'v2.4', 'v2.8', 'v2.9', 'v3.6', 'v6.2', 'v6.8', 'v6.12', 'v6.13', 'v7.1', 'v7.3', 'v7.5', 'v7.6', 'v7.7', 'v7.8', 'v7.9', 'v7.13', 'v7.14', 'v7.15', 'v7.17', 'v8.1', 'v8.2', 'v8.3', 'Geslacht']
#    cols_data = ['v1.4', 'v1.5', 'v1.6', 'v1.7', 'v1.8', 'v1.9', 'v1.10', 'v1.11', 'v2.4', 'v2.8', 'v2.9', 'v3.1', 'v3.3', 'v3.6', 'v6.2', 'v6.5', 'v7.1', 'v7.2', 'v7.3', 'v7.5', 'v7.7', 'v7.8', 'v7.9', 'v7.12', 'v7.13', 'v7.15', 'v7.16', 'Geslacht']
    
    
    X = df_train[cols_data]
    y = df_train[target].replace({0.0: False, 1.0: True})
    print('UNIQUE:', y.unique())
    print(len(X), len(y))
    
    # Bookmark all predictions
    y_pred = []
    
    # Perform kfold CV
    # Apply 5 fold CV
    kf = KFold(n_splits=3) 
    
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
        train_x, test_x = pd.DataFrame(X).loc[train_index], pd.DataFrame(X).loc[test_index]
        train_y, test_y = np.take(y, np.array(train_index)), np.take(y, np.array(test_index))
        print(len(train_x), len(test_x))
        print(len(train_y), len(test_y))

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        param = {
            'objective': "binary:logistic",
            'num_class':1,
            "eval_metric": "auc", #"aucpr",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]), # , "dart"
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            "random_state": 1121218,
            #"n_trees": 50,
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], verbose_eval=0) 
        preds = bst.predict(dtest)
        
        # Create a confusion matrix over all data!
        y_pred.extend(bst.predict(dtest))
    
    #auc_pr = average_precision_score(y, y_pred)
    #error = 1 -balanced_accuracy_score(y, y_pred)#[source]
    aucroc = roc_auc_score(y, y_pred)

    #ppv = precision_score(y, y_pred)
    print(trial._trial_id)
    # accuracy_score
    return aucroc

study = optuna.create_study(direction="maximize")
study.optimize(objective_xgb, n_trials=1000) # test # 2000 # -1 -> bootstrapping (take random 100 samples) , n_jobs=1

print('Best trial: %s' % study.best_trial.number)
print('Performance (accuracy): %s' % study.best_trial.value)
print('Corresponding Parameters: %s' % study.best_trial.params)


import plotly as py
# optuna.visualization.plot_intermediate_values(study)
fig = optuna.visualization.plot_optimization_history(study)
py.offline.plot(fig, filename='hyperparamtuning_optimization_xgb_auc.html', auto_open=False)

#fig = optuna.visualization.plot_contour(study, params=["lambda", "alpha"])
#py.offline.plot(fig, filename='hyperparamtuning_contour_xgb_aucpr.html', image = 'png', auto_open=False)