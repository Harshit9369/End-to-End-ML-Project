import os # to make folders on the current path
import sys # handling custom exceptions. 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

# in any kind of file, we need to import the exception and logger as we need to 
# log the errors and exceptions.

import pandas as pd
from dotenv import load_dotenv
import pymysql # python mysql connector

import pickle
import numpy 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



load_dotenv()
# used to read data from the .env file

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading from SQl DataBase Started!")
    try:
        mydb = pymysql.connect(
            host = host, 
            user = user,
            password = password,
            db = db
        )
        logging.info("Connection Established", mydb)
        df = pd.read_sql_query('Select * from STUDENTS', mydb) # using this method built in from pandas 
        # to read the data from the SQL
        print(df.head())
        
        return df
        
    except Exception as e:
        raise CustomException(e) 

# the below function is used to save the object into a file (pickle) in write mode and the 
# pickle file is saved in the artifacts folder
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
# the below function is used to take all the models and their respective parameters and fits on the data
# 
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            # grid cross-validation
            gs = GridSearchCV(model, para, cv = 3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e, sys)