from config import Config
# data preprocessing

def data_preprocessing(df):
    
    # segregate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
