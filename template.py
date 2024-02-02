# if we run this file, then we get the whole file structure for our required project

import os
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO)

project_name = "mlproject"

list_of_file = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    # above are all the components of the project that we are going to make
    f"src/{project_name}/pipelines/__init__.py", # this line ensures that pipelines is a package
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    # above are the required pipelines for the project
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_file:
    filepath = Path(filepath) # finds the path relative to the project from the list_of_file
    filedir, filename = os.path.split(filepath) # splits the file directory and the file name
    
    if filedir != "": # if the file directory is not empty
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating dictionary: {filedir} for the file {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
            
            
    else:
        logging.info(f"{filename} is already exists")