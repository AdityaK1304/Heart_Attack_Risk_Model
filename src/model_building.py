from config import Config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder




# data building

def data_building(df):
    # define the target variable and features
    X = df.drop(columns =['heart_disease_risk_score'], axis=1)
    y = df['heart_disease_risk_score']
    
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=0.2,
                                                           random_state=42)
    
    # use pipelines for numerical columns

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # use pipelines for categorical columns
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    

    return X_train, X_test, y_train, y_test,numerical_pipeline,categorical_pipeline