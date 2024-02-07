import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    # in this, we want to take the file, then train the model and then save the pickel file 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split Training and Test Input Data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                # not selecting the last column but selecting everything else
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            # below are the models we are going to try and test
            models = {
                "Random Forest": RandomForestRegressor(), 
                "Decision Tree": DecisionTreeRegressor(), 
                "Gradient Boosting": GradientBoostingRegressor(), 
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), 
                "Adaboost Regressor": AdaBoostRegressor()
            }
            # hyperparameter tuning on these models
            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    # "max_depth": [5, 10, 15, 20],
                    # "max_features": ["auto", "sqrt", "log2"],
                },
                "Decision Tree": {
                    # "max_depth": [5, 10, 15, 20],
                    # "max_features": ["auto", "sqrt", "log2"],
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boosting": {
                    # "n_estimators": [100, 200, 300],
                    "learning_rate": [.1, .01, .05, .001],
                    # "max_depth": [5, 10, 15, 20],
                    # "max_features": ["auto", "sqrt", "log2"],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "learning_rate": [.1, .01, .05, .001],
                    # "max_depth": [5, 10, 15, 20],
                    # "max_features": ["auto", "sqrt", "log2"],
                },
                "CatBoosting Regressor": {
                    # "n_estimators": [100, 200, 300],
                    'iterations': [30, 50, 100],
                    'learning_rate': [.1, .01, .05],
                    'depth': [6, 8, 10],
                    # "max_features": ["auto", "sqrt", "log2"],
                },
                "Adaboost Regressor": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "learning_rate": [0.01, 0.1, 0.05, 0.001],
                },
            }
            
            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # to get the model with the best score
            best_model_score = max(sorted(model_report.values()))
            
            # to get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException ("No Best Model found!")
            logging.info(f"Best Model found on both training and testing dataset")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, 
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)

