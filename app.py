from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig
import sys

if __name__ == "__main__":
    logging.info("the execution has started")
    
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        # the above steps have already been executed during the data_ingestion step
        
        data_transformation = DataTransformation()
        #  data_transformation_config = DataTransformationConfig() 
        # this is not needed as we are already calling the DataTransformationConfig function in 
        # DataTransformation class
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        
    except Exception as e: 
        logging.info("Custom Exception")
        raise CustomException(e, sys)