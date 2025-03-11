import optuna
import numpy as np
from scipy.stats import rankdata
from score import score
from preprocess import DataPreprocessor
import pickle

import warnings
warnings.filterwarnings('ignore')

class EnsembleOptimizer:
    def __init__(self, y_true, y_pred, models_dir='gbdt-models', realmlp_dir='realmlp_models', tabm_dir='tabm models', torch_dir='prl-nn-model', prl_dir='prl-nn-mask-model'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.models_dir = models_dir
        self.realmlp_dir = realmlp_dir
        self.tabm_dir = tabm_dir
        self.torch_dir = torch_dir
        self.prl_dir = prl_dir
        self.models = [
        "oof_cat_km_rmse_skf", "pairwise_ranking_oof", "oof_xgb_qcut", "oof_tabm_quantile_skf", "oof_cat_nl_rmse_skf_2",
        "oof_tabm_2", "oof_xgb_tb_rmse", "oof_xgb_ph_rmse_skf", "oof_xgb_twr_rmse", "oof_cat_qcut", "oof_cat_bfhf_rmse_skf",
        "oof_xgb_cox_skf", "oof_cat_tb_rmse_skf_2", "oof_tabm_naf_skf", "oof_cat_qcut_2", "oof_cat_naf_skf", "oof_tabm_bfhf_skf",
        "oof_xgb_nls_rmse", "oof_xgb_mono", "oof_tabm_skf", "oof_xgb_quantile_rmse_skf", "oof_cat_quantile_rmse_2_skf", 
        "oof_cat_bfhf_rmse_skf_2",
        "oof_xgb_naf_rmse_skf", "oof_cat_twr_rmse_skf_2", "oof_cat_nl_rmse_skf", "oof_tabm_prl", "oof_cat_aft_skf", "oof_pred_nn",
        "oof_xgb_quantile_mae", "oof_xgb_bfhf_rmse_skf", "oof_cat_tb_rmse_skf", "oof_tabm_rank", "oof_xgb_km_rmse_skf",
        "oof_cat_twr_rmse_skf", "oof_cat_cox_2_skf", "oof_cat_cox_skf", "oof_cat_naf_2_skf","oof_cat_km_rmse_2_skf",
        "oof_cat_quantile_rmse_skf"
        ]

        self.model_predictions = {}
        
    def load_models(self):
        """Load all model predictions from pickle files"""
        for model_name in self.models:
            try:
                # Determine which directory to use based on model type
                if model_name.startswith("oof_tabm"):
                    model_path = f'{self.tabm_dir}/{model_name}.pkl'
                elif model_name.startswith("oof_pred_nn"):
                    model_path = f'{self.torch_dir}/{model_name}.pkl'
                elif model_name.startswith("pairwise_ranking_oof"):
                    model_path = f'{self.prl_dir}/{model_name}.pkl'
                elif model_name.startswith("oof_realmlp") or model_name.startswith("oof_resnet"):
                    model_path = f'{self.realmlp_dir}/{model_name}.pkl'
                else:
                    model_path = f'{self.models_dir}/{model_name}.pkl'
                
                with open(model_path, 'rb') as f:
                    self.model_predictions[model_name] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Could not find file for model {model_name}")
    
    def objective(self, trial):
        """Optimization objective function for Optuna"""
        selected_models = []
        predictions = np.zeros_like(rankdata(self.model_predictions[self.models[0]]))

        for model in self.models:
            if trial.suggest_categorical(model, [0, 1]):
                selected_models.append(model)
                predictions += rankdata(self.model_predictions[model])

        if not selected_models:
            return -999

        y_pred_copy = self.y_pred.copy()
        y_pred_copy["prediction"] = predictions
        return score(self.y_true.copy(), y_pred_copy, "ID")

    def optimize(self, n_trials=1000):
        """Run the optimization process"""
        study = optuna.create_study(direction="maximize")
        good_trials = []  # List to store good trials

        def callback(study, trial):
            if trial.value > 0.690:
                good_trials.append({
                    'score': trial.value,
                    'models': [m for m in self.models if trial.params[m] == 1]
                })

        study.optimize(self.objective, n_trials=n_trials, callbacks=[callback])
        
        # Sort good trials by score in descending order
        good_trials.sort(key=lambda x: x['score'], reverse=True)
        
        best_models = [m for m in self.models if study.best_trial.params[m] == 1]
        return study.best_value, best_models, good_trials

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
    import pickle

    warnings.filterwarnings("ignore")

    # Assume that main() and score() functions are defined elsewhere.
    train_features, test_features, y_true, y_pred = main()

    # Fixed base models and their fixed weights (provided by you).
    base_model_names = [
        'oof_cat_km_rmse_skf', 'pairwise_ranking_oof', 'oof_tabm_bfhf_skf', 'oof_xgb_quantile_rmse_skf', 'oof_tabm_prl', 'oof_pred_nn', 'oof_cat_km_rmse_2_skf', 'oof_xgb_mono',
        'oof_cat_naf_2_skf', 'oof_cat_bfhf_rmse_skf_2', 'oof_cat_bfhf_rmse_skf'
    ]
    fixed_base_weights = np.array([1, 10, 1, 1, 10, 9, 2, 6, 3, 1, 8])

    # Pool of additional models, which will be included only if their weight > 0.
    additional_model_names = [
        "oof_xgb_qcut", "oof_xgb_tb_rmse",
        "oof_xgb_ph_rmse_skf", "oof_xgb_twr_rmse",
        "oof_xgb_naf_rmse_skf", 
         "oof_xgb_quantile_mae", "oof_xgb_bfhf_rmse_skf",
         "oof_xgb_km_rmse_skf", 
         "oof_xgb_cox_skf",
         "oof_realmlp", "oof_resnet",
          "oof_lgb_km_rmse", "oof_lgb_naf_rmse", "oof_lgb_km_rmse"
    ]

    # oof_xgb_cox_skf



    # Pre-load predictions for all models (base and additional) and convert them to their rank order.
    model_preds = {}

    def get_model_path(model_name):
        if model_name.startswith("oof_tabm"):
            return f'tabm models/{model_name}.pkl'
        elif model_name.startswith("oof_pred_nn"):
            return f'prl-nn-model/{model_name}.pkl'
        elif model_name.startswith("pairwise_ranking_oof"):
            return f'prl-nn-mask-model/{model_name}.pkl'
        elif model_name.startswith("oof_realmlp") or model_name.startswith("oof_resnet"):
            return f'realmlp_models/{model_name}.pkl'
        else:
            return f'gbdt-models/{model_name}.pkl'

    for model_name in base_model_names + additional_model_names:
        with open(get_model_path(model_name), 'rb') as f:
            pred = pickle.load(f)
            model_preds[model_name] = rankdata(pred)

    # Define the objective function for Optuna to optimize weights for additional models.
    def objective(trial):
        # For each additional model, allow its weight to be in [0, 10] (0 means the model is dropped).
        additional_weights = np.array([
            trial.suggest_int(f"w_add_{model}", 0, 3) for model in additional_model_names
        ])
        
        # The complete weight vector is the fixed base weights concatenated with the additional weights.
        full_weights = np.concatenate([fixed_base_weights, additional_weights])
        
        # Retrieve predictions in the same order.
        base_preds_list = [model_preds[m] for m in base_model_names]
        additional_preds_list = [model_preds[m] for m in additional_model_names]
        all_preds = np.array(base_preds_list + additional_preds_list)
        
        # Compute the ensemble prediction as the weighted sum of ranked predictions.
        ensemble_preds = np.dot(full_weights, all_preds)
        
        # Update y_pred with the ensemble prediction and calculate the score.
        y_pred_temp = y_pred.copy()
        y_pred_temp["prediction"] = ensemble_preds
        score_value = score(y_true.copy(), y_pred_temp.copy(), "ID")
        return score_value

    # Run the Optuna study.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value

    # Identify additional models selected (those with a nonzero weight).
    selected_additional = [
        model for model in additional_model_names if best_params.get(f"w_add_{model}", 0) > 0
    ]
    best_ensemble_models = base_model_names + selected_additional

    print('Best ensemble saved.')
    print("Optuna best parameters:", best_params)
    print("Optuna best score:", best_score)
    print("Models used in the best ensemble:")
    for model in best_ensemble_models:
        print(model)
