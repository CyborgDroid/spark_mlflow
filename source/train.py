#%%
# Import packages and data
import pyspark.sql.functions as F
import pyspark.sql.types as T
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
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

#%%
# Gradient-boosted tree and LSVC grid searches

GBT_params = {
    'maxDepth': [5],
    'maxBins': [48, 64],
    'maxIter': [25],
    'stepSize': [0.1, 0.15]
}

LSVC_params = {
    'standardization': [True],
    'aggregationDepth': [2, 5, 10],
    'regParam': [0.001, 0.01],
    'maxIter': [25],
    'tol': [1e-6, 1e-4]
}

MLP_params = {
    'layers': [[123, 10, 4, 2]],
    'blockSize': [5, 10],
    'stepSize': [0.001, 0.01],
    'maxIter': [25],
    'tol': [1e-6, 1e-4]
}

LR_params = {
    'standardization': [True],
    'aggregationDepth': [5, 10],
    'regParam': [0.001, 0.01],
    'maxIter': [25],
    'threshold':[0.5],
    'elasticNetParam': [0.0, 0.5, 1],
    'tol': [1e-6, 1e-4]
}

RandomForest_params = {
    'maxDepth': [5],
    'maxBins': [48],
    'minInfoGain': [0.0, 0.05],
    'impurity': ['gini', 'entropy']
}

grid_search_results = SparkMLBinaryClassifier(
    trainingData,
    testData,
    evaluator='MulticlassClassificationEvaluator',
    label_col='label',
    features_col='features',
    kfolds=3,
    GBT_params=GBT_params,
    LSVC_params=LSVC_params,
    MLP_params=MLP_params,
    LR_params=LR_params,
    RandomForest_params=RandomForest_params)

#%%
# Random Search or GA
# MLP_params = {
#     'num_hidden_layers': range(1,5),
#     'first_hidden_layer_size': range(2,21,4),
#     'blockSize': [2, 5, 10],
#     'stepSize': [0.001, 0.01],
#     'maxIter': [25],
#     'tol': [1e-6, 1e-4]
# }

#%%
#explain params

param_map = grid_search_results.models['GBT'].bestModel.stages[0].extractParamMap()

for p in param_map:
    print(p.name, param_map.get(p.name))

#%%
