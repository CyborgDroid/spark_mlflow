#%%

import pyspark.sql.functions as F
import pyspark.sql.types as T
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.functions import SparkMethods, DataLoader

spark = SparkMethods.get_spark_session()
#%%
df = DataLoader.load_data("data/adult.test")
df = df.where(F.col('age').isNotNull())
df.show()

#%%
# Load vectorizer and model
import mlflow.spark
vectorizer = mlflow.spark.load_model('mlruns/2/36df65d8b1f14cef9b5f2c1bf4af736d/artifacts/vectorizer')
model = mlflow.spark.load_model('mlruns/1/c17bf9c49b884bd69b7ff24fe01efbbd/artifacts/bestGBT') 

#%%
# vectorize and predict
transformed_df = vectorizer.transform(df)
transformed_df = model.transform(transformed_df)
transformed_df.show(20)

#%%[markdown]
### Check predictions ###
#%%
# Clean income column in test data that contains a period after the income
transformed_df = transformed_df.withColumn('income', F.regexp_replace(F.col('income'), r'\.', ''))
transformed_df.show()

#%%
# Index label (income)
label_vectorizer = mlflow.spark.load_model('mlruns/2/36df65d8b1f14cef9b5f2c1bf4af736d/artifacts/label_vectorizer') 
transformed_df = label_vectorizer.transform(transformed_df)
transformed_df.show()

#%%
# Check accuracy
SparkMethods.get_MultiClassMetrics(transformed_df, data_type='val')

#%%
#%% get data for testing:
test_data = df.limit(2).toPandas().to_json(orient='split')
print(test_data)

#%%
