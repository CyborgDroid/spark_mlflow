#%%
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LinearSVC, MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.feature import ChiSqSelector, StringIndexer, OneHotEncoderEstimator, VectorAssembler, MinMaxScaler, IndexToString, SQLTransformer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml import Pipeline
import os
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from datetime import date
from source.functions import SparkMethods, DataLoader, SparkMLBinaryClassifierRandomSearch

#%%
spark = SparkMethods.get_spark_session()

df = DataLoader.load_data("data/adult.data")

#%%

categorical_cols = [
    'workclass', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

scaling_cols = [
    'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'
]
#%%
#create vectorizer model and transform df
vectorizer, label_vectorizer, transformed_df = SparkMethods.vectorizer(
    df,
    labels_to_vectorize={'income': 'OneHotEncoderEstimator'},
    CategoricalCols=categorical_cols,
    MinMaxCols=scaling_cols)
transformed_df.show()

#%%
# Split the data into training and test sets (30% held out for testing)
strata_cols = ['income', 'sex']
trainingData, testData = SparkMethods.train_test_split(transformed_df,
                                                       strata_cols,
                                                       trainRatio=0.7,
                                                       show_summary=True)

#%%
models = SparkMLBinaryClassifierRandomSearch(
    trainingData,
    testData,
    random_grid_size=8,
    kfolds=3,
    grid_params={
        'GBTClassifier': {
            'maxDepth': [3, 5, 7, 9],
            'maxBins': [8, 16, 32, 48, 64],
            'maxIter': [25, 50, 75],
            'stepSize': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        },
        'LinearSVC': {
            'standardization': [True, False],
            'aggregationDepth': [2, 5, 7, 10],
            'regParam': [0.001, 0.01, 0.1, 1.0, 10.0],
            'maxIter': [25, 50, 75],
            'tol': [1e-06, 0.0001, 0.01]
        },
        'MultilayerPerceptronClassifier': {
            'num_hidden_layers': range(1, 5),
            'first_hidden_layer_size': range(2, 21, 4),
            'blockSize': [2, 5, 10],
            'stepSize': [0.001, 0.01, 0.1],
            'maxIter': [25, 50, 75],
            'tol': [1e-06, 0.0001, 0.01]
        },
        'LogisticRegression': {
            'standardization': [True, False],
            'aggregationDepth': [2, 5, 7, 10],
            'regParam': [0.001, 0.01, 0.1, 1.0, 10.0],
            'maxIter': [25, 50, 75],
            'threshold': [0.4, 0.5, 0.6],
            'elasticNetParam': [0.0, 0.25, 0.5, 0.75, 1.0],
            'tol': [1e-06, 0.0001, 0.01]
        },
        'RandomForestClassifier': {
            'maxDepth': [3, 5, 7, 9],
            'maxBins': [8, 16, 32, 48, 64],
            'minInfoGain': [0.0, 0.05, 0.1],
            'impurity': ['gini', 'entropy']
        }
    })

#%%
