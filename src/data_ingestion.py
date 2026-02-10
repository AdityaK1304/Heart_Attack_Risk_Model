from config import Config
import pandas as pd
import logging

# Data ingestion

def data_ingestion():
    try:
        #load the data
        df = pd.read_csv(r'C:\Heart_Attack_Risk_Model\data\raw\cardiovascular_risk_dataset.csv')
        logging.info("Data ingestion successful.")
    except:
        logging.error("Data ingestion failed.")
    return df


