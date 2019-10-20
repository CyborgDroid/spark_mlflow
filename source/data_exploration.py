#%%
# Import packages and data
import pyspark.sql.functions as F
import pyspark.sql.types as T
from source.functions import SparkMethods, DataLoader
import os
import io

spark = SparkMethods.get_spark_session()

df = DataLoader.load_data("data/adult.data")

df.show()

#%%
#General overview of data
SparkMethods.describe(df)

#%%
profile = SparkMethods.create_pandas_profile(df, 'overview')

#%%
#%%
categorical_cols = [
    'workclass'
    , 'education'
    , 'education-num'
    , 'marital-status'
    , 'occupation'
    , 'relationship'
    , 'race'
    , 'sex'
    , 'native-country'
]