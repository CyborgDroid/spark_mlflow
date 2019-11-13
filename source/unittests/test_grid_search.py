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
from source.functions import SparkMethods, DataLoader, SparkMLBinaryClassifier

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
trainingData, testData = SparkMethods.train_test_split(
    transformed_df,
    strata_cols,
    trainRatio=0.7,
    show_summary=True)

#%%
models = SparkMLBinaryClassifier(trainingData, testData, kfolds=3, 
    GBT_params = {
        'maxDepth': [5],
        'maxBins': [48],
        'maxIter': [25],
        'stepSize': [0.1, 0.15]
    },
    LSVC_params = {
        'standardization': [True],
        'aggregationDepth': [5, 10],
        'regParam': [0.001, 0.01],
        'maxIter': [25],
        'tol': [1e-6, 1e-4]
    },
    MLP_params = {
        'layers': [[123, 10, 4, 2]],
        'blockSize': [5, 10],
        'stepSize': [0.001, 0.01],
        'maxIter': [25],
        'tol': [1e-6, 1e-4]
    },
    LR_params = {
        'standardization': [True],
        'aggregationDepth': [5, 10],
        'regParam': [0.001, 0.01],
        'maxIter': [25],
        'threshold':[0.5],
        'elasticNetParam': [0.0, 0.5, 1],
        'tol': [1e-6, 1e-4]
    },
    RandomForest_params = {
        'maxDepth': [5],
        'maxBins': [48],
        'minInfoGain': [0.0, 0.05],
        'impurity': ['gini', 'entropy']
    }

    )

#%%
