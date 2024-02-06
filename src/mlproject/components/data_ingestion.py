# Reading Data from the DataBase then train-test-split. 
import os # to make folders on the current path
import sys # handling custom exceptions. 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
# in any kind of file, we need to import the exception and logger as we need to 
# log the errors and exceptions.

import pandas as pd
from src.mlproject.utils import read_sql_data

from sklearn.model_selection import train_test_split

from dataclasses import dataclass 
# the input parameters will get initialised easily and fast using this 

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            # reading the data from SQL
            # df = read_sql_data() -> once the data has been read through sql, we don't need to do this 
            df = pd.read_csv(os.path.join('notebook/data', 'raw.csv'))
            # the sql data has been read and saved into a csv file in the notebook/data folder which 
            # we are reading here.
            
            logging.info("Reading completed SQL DataBase")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion Completed!")
            
            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)