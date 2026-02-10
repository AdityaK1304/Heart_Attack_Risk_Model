from config import Config
from model_evalution import model_evaluation
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import data_building
from src.model_evalution import model_evaluation

def main():
    df = data_ingestion()
    print(df.head())
    data = data_preprocessing(df)
    X_train, X_test, y_train, y_test,numerical_pipeline,categorical_pipeline = data_building(data)
    model_evaluation(X_train, X_test, y_train, y_test,numerical_pipeline,categorical_pipeline)


if __name__ == "__main__":
    main()