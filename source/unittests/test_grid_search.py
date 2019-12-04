#%%
from source.functions import SparkMethods, DataLoader, SparkMLBinaryClassifierGridSearch

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
# run: 8 GBT, 8 LSVC, 8 MLPs, 8 LRs, and 8 RandomForests
models = SparkMLBinaryClassifierGridSearch(trainingData, testData, labelCol='label', featuresCol='features', kfolds=3,
    grid_params={
        'GBTClassifier': {
            'maxDepth': [5, 7],
            'maxBins': [32,48],
            'maxIter': [25],
            'stepSize': [0.1, 0.15]
        },
        'LinearSVC' : {
            'standardization': [True],
            'aggregationDepth': [5, 10],
            'regParam': [0.001, 0.01],
            'maxIter': [25],
            'tol': [1e-6, 1e-4]
        },
        'MultilayerPerceptronClassifier': {
            'layers': [[123, 10, 4, 2]],
            'blockSize': [5, 10],
            'stepSize': [0.001, 0.01],
            'maxIter': [25],
            'tol': [1e-6, 1e-4]
        },
        'LogisticRegression': {
            'standardization': [True],
            'aggregationDepth': [5, 10],
            'regParam': [0.001, 0.01],
            'maxIter': [25],
            'threshold':[0.5],
            'elasticNetParam': [0.0],
            'tol': [1e-6, 1e-4]
        },
        'RandomForestClassifier': {
            'maxDepth': [5],
            'maxBins': [32, 48],
            'minInfoGain': [0.0, 0.05],
            'impurity': ['gini', 'entropy']
        }
    })

#%%
