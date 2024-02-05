import os # to make folders on the current path
import sys # handling custom exceptions. 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
# in any kind of file, we need to import the exception and logger as we need to 
# log the errors and exceptions.

import pandas as pd
from dotenv import load_dotenv
import pymysql # python mysql connector


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