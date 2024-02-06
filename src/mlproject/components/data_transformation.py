import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # for handling missing data

from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging 

from src.mlproject.utils import save_object

import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    # setting a path for when feature engineering takes place, we will be saving that feature engineering model into 
    # a pkl (pickle file) and we will be saving it in the artifacts folder

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # if in case we get missing values or something like that in data then:
            num_pipeline = Pipeline(steps=[ # for handling missing values or something like
            # that in numerical columns
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[ # for handling missing values or something like that in 
            # categorical columns
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()), 
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
    
    
            
        except Exception as e:
            raise CustomException(e, sys)
        
    # the output of data_ingestion (train_path and test_path) should go into this method
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading the Train and Test file")
            
            preprocessing_obj = self.get_data_transformer_object()
            # we are calling the data_transformer_object function using the above method
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            # dividing the train dataset into independent and dependent features
            
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # dividing the test dataset into independent and dependent features
            
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying the preprocessor on the train and test dataset")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            # np.c_ -> np.concatenate -> combines both the arrays into one array column-wise
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved Preprocessing Object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            # the above function is taken from the utils file 
            # and is used to save the object into a file (pickle) in write mode and the
            # path is given in the data_transformation_config class
            
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
        except Exception as e:
            raise CustomException (sys, e)