import optuna
import numpy as np
from scipy.stats import rankdata
from score import score
from preprocess import DataPreprocessor
import pickle

import warnings
warnings.filterwarnings('ignore')

class DataManager:
    def __init__(self):
        self.train_path = r"C:\Projects\CIBMTR\data\cibmtr\train.csv"
        self.test_path = r"C:\Projects\CIBMTR\data\cibmtr\test.csv"
    
    def load_and_preprocess(self):
        """Load and preprocess the data"""
        preprocessor = DataPreprocessor(self.train_path, self.test_path)
        train_processed, test_processed = preprocessor.process()
        
        # Extract target variables and IDs
        y_true = train_processed[["ID", "efs", "efs_time", "race_group"]].copy()
        y_pred = train_processed[["ID"]].copy()
        
        # Remove these columns from training data
        train_features = train_processed.drop(["ID", "efs", "efs_time", "race_group"], axis=1)
        test_features = test_processed.drop(["ID"], axis=1)
        
        return train_features, test_features, y_true, y_pred

def main():
    # Initialize data manager and load data
    data_manager = DataManager()
    train_features, test_features, y_true, y_pred = data_manager.load_and_preprocess()
    
    # # Initialize and run ensemble optimizer with y_true and y_pred
    # optimizer = EnsembleOptimizer(y_true=y_true, y_pred=y_pred)
    # optimizer.load_models()
    # best_score, best_models, good_trials = optimizer.optimize()
    
    # print(f"\nBest Ensemble Score: {best_score}")
    # print(f"Best Model Subset: {best_models}")
    # print(f'Length of best models: {len(best_models)}')
    
    return train_features, test_features, y_true, y_pred

if __name__ == "__main__":

    import numpy as np
    import optuna
    from scipy.stats import rankdata
    import warnings
    from scipy.optimize import minimize
    warnings.filterwarnings("ignore")

    train_features, test_features, y_true, y_pred = main()


    with open("C:\Projects\CIBMTR\data\cibmtr\ensemble_oof_preds_1.pkl", "rb") as f:
        ensemble_oof_preds_1 = pickle.load(f)
    with open("C:\Projects\CIBMTR\data\cibmtr\ensemble_oof_preds_2.pkl", "rb") as f:
        ensemble_oof_preds_2 = pickle.load(f)
    with open("C:\Projects\CIBMTR\data\cibmtr\ensemble_oof_preds_3.pkl", "rb") as f:
        ensemble_oof_preds_3 = pickle.load(f)

    oof_preds = [
        ensemble_oof_preds_1,
        ensemble_oof_preds_2,
        ensemble_oof_preds_3
        ]
    
    ranked_oof_preds = np.array([rankdata(p) for p in oof_preds])

    def objective(trial):
        """
        Objective function for Optuna.
        Each trial suggests a set of weights and computes the ensemble
        predictions and the resulting score.
        """
        # Suggest weights for each model; adjust the range as needed.
        w1 = trial.suggest_int("w1", 1, 10)
        w2 = trial.suggest_int("w2", 1, 10)
        w3 = trial.suggest_int("w3", 1, 10)
        
        weights = np.array([w1, w2, w3])
        
        # Compute ensemble predictions on the training (oof) set using the weights
        ensemble_oof_preds = np.dot(weights, ranked_oof_preds)
        
        # Create a copy of y_pred and assign predictions
        y_pred_temp = y_pred.copy()
        y_pred_temp["prediction"] = ensemble_oof_preds
        
        # Calculate the score (assuming a higher score is better)
        m = score(y_true.copy(), y_pred_temp.copy(), "ID")
    
    # Return the score directly because we'll configure Optuna to maximize it.
        return m


    study = optuna.create_study(direction="maximize")

    # Optimize the objective function. Adjust n_trials as needed.
    study.optimize(objective, n_trials=300)

    # Get the best weights and best score
    best_weights = np.array([
        study.best_params["w1"],
        study.best_params["w2"],
        study.best_params["w3"]

    ])
    best_score = study.best_value

    print("Optimal weights:", best_weights)
    print("Best score:", best_score)

    # Optionally, generate test predictions using the optimal weights:
    # best_weights = np.array([1, 2, 1, 2, 10, 6, 1, 5])

    ensemble_oof_preds = np.dot(best_weights, ranked_oof_preds)

    y_pred["prediction"] = ensemble_oof_preds
    m = score(y_true.copy(), y_pred.copy(), "ID")
    print(f"\nOverall CV for ENSEMBLE =",m)

