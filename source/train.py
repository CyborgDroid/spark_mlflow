#%%
# Import packages and data
import pyspark.sql.functions as F
import pyspark.sql.types as T
import mlflow
import os, sys, io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.functions import SparkMethods, DataLoader
# from mlflow.exceptions import MlflowException
# from mlflow.tracking import MlflowClient
# from datetime import date
# today = date.today()

# experimentPath = today.strftime("%Y%m%d")

# try:
#     experimentID = mlflow.create_experiment(experimentPath)
#     print('created new MLFlow Experiment')
# except MlflowException:
#     print('Using existing MLFlow Experiment')
#     experimentID = MlflowClient().get_experiment_by_name(
#         experimentPath).experiment_id
#     mlflow.set_experiment(experimentPath)

spark = SparkMethods.get_spark_session()

df = DataLoader.load_data("data/adult.data")

#%%

log_metrics = {}

categorical_cols = [
    'workclass', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

scaling_cols = [
    'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'
]

#%%
#create vectorizer model and transform df
vectorizer, transformed_df = SparkMethods.vectorizer(
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
# Gradient-boosted tree classifier pipeline

grid_params = {
    'maxDepth': [1,5],
    'maxBins': [8],
    'maxIter': [10],
    'stepSize': [0.15]
}
print(grid_params)
cv_model, train_df, test_df = SparkMethods.grid_search_GBT(
                                            trainingData, 
                                            testData,
                                            evaluator='MulticlassClassificationEvaluator',
                                            label_col='label',
                                            features_col='features',
                                            grid_params=grid_params)


#%%

cv_model.bestModel.extractParamMap()

#%%
cv_model.explainParams()
#%%
