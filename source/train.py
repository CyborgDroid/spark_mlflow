#%%
# Import packages and data
import pyspark.sql.functions as F
import pyspark.sql.types as T
import mlflow
import os, sys, io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.functions import SparkMethods, DataLoader
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from datetime import date
today = date.today()

experimentPath = today.strftime("%Y%m%d")

try:
    experimentID = mlflow.create_experiment(experimentPath)
except MlflowException:
    experimentID = MlflowClient().get_experiment_by_name(
        experimentPath).experiment_id
    mlflow.set_experiment(experimentPath)

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
    'maxDepth': [1],
    'maxBins': [32],
    'maxIter': [20],
    'stepSize': [0.15]
}

best_model, submodels, param_map, best_df = SparkMethods.grid_search_GBT(
    trainingData,
    label_col='label',
    features_col='features',
    grid_params=grid_params)

#%%
print(len(submodels))
print(len(submodels[0]))

#%%

import mlflow.spark
for i, m in enumerate(submodels):
    mlflow.spark.log_model(m, 'gbt_model_' + str(i + 1))
#%%
best_df.select('income', 'label', 'probability', 'predicted_label').show()
#%%

from pyspark.mllib.evaluation import MulticlassMetrics


def calculate_metrics(df, model):
    transformed_df = model.transform(df)
    p = 'label'

    print('\nRESULTS FOR ' + p.upper())
    predictionAndLabels = transformed_df.select(
        F.col('predicted_label').cast(T.FloatType()),
        F.col(p).cast(T.FloatType())).rdd

    metrics = MulticlassMetrics(predictionAndLabels)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    # print("Summary Stats")
    # print("Precision = %s" % precision)
    # print("Recall = %s" % recall)
    # print("F1 Score = %s" % f1Score)

    # # Statistics by class
    # labels = [0.0, 1.0]
    # for label in sorted(labels):
    #     print("Class %s precision = %s" % (label, metrics.precision(label)))
    #     print("Class %s recall = %s" % (label, metrics.recall(label)))
    #     print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    #     mlflow.log_metric('Train_recall_' + str(label),  float(metrics.recall(label)))
    #     mlflow.log_metric('Train_precision_' + str(label),  float(metrics.precision(label)))
    #     mlflow.log_metric('Train_F1_score_' + str(label),  float(metrics.fMeasure(label, beta=1.0)))

    # # Weighted stats
    # print("Weighted recall = %s" % metrics.weightedRecall)
    # print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    # print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    # print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)


#%%
print('Best Model')
calculate_metrics(testData, best_model)

#%%
for i, fold in enumerate(submodels):
    print('\n***Fold ' + str(i + 1) + '***\n')
    for x, model in enumerate(fold):
        print('\nModel ' + str(x + 1))
        calculate_metrics(testData, model)

#%%
from pyspark.ml.classification import GBTClassifier
[x._java_obj.extractParamMap() for x in best_model.stages]
#%%
best_model.featureImportances

#%%
param_map

#%%
best_df.show()

#%%

sc = spark.sparkContext
print(sc.getConf().getAll())

#%%
