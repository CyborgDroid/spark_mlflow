import unicodedata
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import re
from configparser import RawConfigParser
from pathlib import Path
import pandas_profiling
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import ChiSqSelector, StringIndexer, OneHotEncoderEstimator, VectorAssembler, MinMaxScaler, IndexToString
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml import Pipeline
import os
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import mlflow
from datetime import date
from pathlib import Path

class GeneralMethods:
    @staticmethod
    def get_project_root() -> Path:
        """Returns project root folder."""
        return str(Path(__file__).parent.parent)

class SparkMethods:
    """General Spark methods including Exploratory Data Analysis, ETL, MLlib, and MLFlow.
    """
    @staticmethod
    def get_spark_session():
        """Get databricks or current spark session or create a new one

        Returns:
            pyspark.sql.session.SparkSession
        """
        return SparkSession.builder\
            .appName("spark_mlflow")\
            .master("local[*]")\
            .config("spark.sql.execution.arrow.enabled", "true")\
            .config("spark.sql.execution.arrow.fallback.enabled", "true")\
            .getOrCreate()

    @staticmethod
    def create_pandas_profile(spark_df, report_title):
        """
        Profile a spark dataframe by converting it to pandas.

        Arguments:
            spark_df {pyspark.sql.dataframe.DataFrame} -- df to profile
            report_title {str} -- displayed on the report and used to create the file name
        """
        # normalize col_names for profiling
        import re
        import string
        pattern = re.compile(r'[\W]+', re.UNICODE)

        for c in spark_df.columns:
            c1 = c.replace(' ', '_')
            c1 = pattern.sub('', c1)
            spark_df = spark_df.withColumnRenamed(c, c1)

        file_name = pattern.sub('', report_title.replace(' ', '_'))

        pandas_df = spark_df.toPandas()
        pandas_df.head()
        profile = pandas_df.profile_report(title=report_title)
        profile.to_file(output_file=Path("./reports/" + file_name + ".html"))
        return profile

    @staticmethod
    def describe(df, max_cols_to_show=100):
        """ Describe spark df, display nicely, and add missing data column

        Arguments:
            df {pyspark.sql.dataframe.DataFrame}

        Keyword Arguments:
            max_cols_to_show {int} -- (default: {100})
        """
        describe_df = df.describe()

        # Add percent ? as nulls
        def is_question_mark(c):
            return F.sum(F.when(F.trim(F.col(c)) == '?',
                                1).otherwise(0)).alias(c)
        questions_df = df.groupby(F.lit('question_marks').alias('summary'))\
            .agg(*[is_question_mark(c) for c in df.columns])
        describe_df = describe_df.union(questions_df)

        # Add number of unique values:

        distinct_df = df.groupby(F.lit('unqiue_values').alias('summary'))\
            .agg(*[F.countDistinct(c) for c in df.columns])
        describe_df = describe_df.union(distinct_df)

        # convert to pandas for pretty_printing
        summary_pdf = describe_df.toPandas().set_index('summary').T
        total = df.count()
        print('Total rows:', total)
        summary_pdf['count'] = summary_pdf['count'].astype(int)
        summary_pdf['question_marks'] = summary_pdf['question_marks'].astype(
            int)
        summary_pdf['perc_null'] = round(
            (total - summary_pdf['count'] + summary_pdf['question_marks']) /
            total, 5)
        perfect_pdf = summary_pdf[summary_pdf['perc_null'] == 0]
        # print('COLUMNS WITH NO MISSING VALUES:')
        # print(perfect_pdf.to_string())
        missing_pdf = summary_pdf[summary_pdf['perc_null'] > 0]
        # print('COLUMNS WITH MISSING VALUES:')
        # print(missing_pdf.to_string())

        return summary_pdf

    @staticmethod
    def train_test_split(df,
                         strata_columns,
                         row_id='monotonically_increasing_id',
                         trainRatio=0.8,
                         show_summary=False):
        """Stratified train/test split based on certain columns. Usually only the target label.
        Can also be used to ensure other stratifications are used like gender or race. 

        Arguments:
            df {pyspark.sql.dataframe.DataFrame} 
            row_id {string} -- Unique ID per row, if none is provided one will be created assuming all rows are unique
            strata_columns [string] -- List of column names to stratify split

        Keyword Arguments:
            trainRatio {float} -- percent to split training (default: {0.8})
            show_summary {bool} -- print summary to console (default: {False})

        Raises:
            ValueError: row_id must be unique

        Returns:
            [pyspark.sql.dataframe.DataFrame] -- train_df and test_df
        """

        # Create a row_id or check if provided row_id is unique:
        if row_id == 'monotonically_increasing_id':
            df = df.withColumn("monotonically_increasing_id",
                               F.monotonically_increasing_id())
        elif df.groupby(row_id).agg(F.count(row_id).alias('count')).where(
                F.col('count') > 1).count():
            raise ValueError('invalid row_id, it must be unique')
        # create strata column
        df = df.withColumn(
            'strata',
            F.concat_ws(
                ',', *[F.col(x).cast(T.StringType()) for x in strata_columns]))
        # loop through all the strata
        strata = [
            s['strata'] for s in df.select('strata').distinct().collect()
        ]
        for not_first, s in enumerate(strata):
            strata_df = df.where(F.col('strata') == s)
            test_count = int(strata_df.count() * trainRatio)
            train_df_0 = strata_df.limit(test_count)
            test_df_0 = strata_df.join(train_df_0, [row_id], how='leftanti')
            if not_first:
                train_df = train_df.union(train_df_0)
                test_df = test_df.union(test_df_0)
            else:
                train_df = train_df_0
                test_df = test_df_0
        if show_summary:
            print('Train/test strata:')
            train_df.groupby('strata').agg(F.count('strata').alias('train_count'))\
                .join(test_df.groupby('strata').agg(F.count('strata').alias('test_count')), ['strata'], how='outer')\
                .withColumn('test_%', F.round(F.col('test_count')/(F.col('test_count')+F.col('train_count')), 3))\
                .show()
        return train_df, test_df

    @staticmethod
    def remove_accents(col_name):
        """Removes accents for spark DF text column

        Example:
            df.withColumn('clean_text', remove_accents('text'))

        Arguments:
            col_name {str} -- column name with text

        Returns:
            str -- text without accents
        """
        def make_trans():
            matching_string = ""
            replace_string = ""

            for i in range(ord(" "), sys.maxunicode):
                name = unicodedata.name(chr(i), "")
                if "WITH" in name:
                    try:
                        base = unicodedata.lookup(name.split(" WITH")[0])
                        matching_string += chr(i)
                        replace_string += base
                    except KeyError:
                        pass

            return matching_string, replace_string

        def clean_text(c):
            matching_string, replace_string = make_trans()
            return F.translate(F.regexp_replace(c, r"\p{M}", ""),
                               matching_string, replace_string).alias(c)

        return clean_text(col_name)

    @staticmethod
    def vectorizer(df,
                   labels_to_vectorize={'label': 'OneHotEncoderEstimator'},
                   CategoricalCols=[],
                   MinMaxCols=[]):
        """
        Vectorizes categorical columns and numerical columns that can be scaled.
        Currently only supports one label
        Supported encoders:
            OneHotEncoderEstimator
            MinMaxScaler

        Arguments:
            df {pyspark.sql.dataframe.DataFrame} 

        Keyword Arguments:
            label_cols {list} --  (default: {['label']})
            CategoricalCols {list} --  Converted using OneHotEncoderEstimator (default: {[]})
            MinMaxCols {list} -- [Scaled using MinMaxScaler] (default: {[]})

        Returns:
            [vectorizer, transformed_df] -- Returns vectorizer model and transformed DF
        """
        # MLFlow logs
        import mlflow
        today = date.today()
        experimentPath = today.strftime("%Y%m%d")
        try:
            experimentID = mlflow.create_experiment(experimentPath)
        except MlflowException:
            experimentID = MlflowClient().get_experiment_by_name(
                experimentPath).experiment_id
            mlflow.set_experiment(experimentPath)

        # with mlflow.start_run(experiment_id=experimentID,run_name='vectorizer', nested=True):
        params = {'CategoricalCols': CategoricalCols, 'MinMaxCols': MinMaxCols}
        params.update(labels_to_vectorize)
        mlflow.log_params(params)
        # create all requried column names
        cat_index_cols = [c + '_index' for c in CategoricalCols]
        cat_vec_cols = [c + '_vec' for c in CategoricalCols]
        scaled_cols = [c + '_scaled' for c in MinMaxCols]

        # Scaling
        scaling_assembler = VectorAssembler(inputCols=MinMaxCols,
                                            outputCol="scaling_features")
        scaler = MinMaxScaler(inputCol="scaling_features",
                              outputCol='scaled_features')

        # Index labels, adding metadata to the label column.
        def index(cat_col):
            cat_index_col = cat_col + '_index'
            indexer = StringIndexer(inputCol=cat_col, outputCol=cat_index_col)
            return indexer

        # Categorical encoding
        encoder = OneHotEncoderEstimator(inputCols=cat_index_cols,
                                         outputCols=cat_vec_cols,
                                         dropLast=False)

        assembler = VectorAssembler(inputCols=cat_vec_cols +
                                    ['scaled_features'],
                                    outputCol='features')

        label_indexers = []
        # label vectorization: GBT expects column called 'label' so having multiple labels is not possible
        for label in labels_to_vectorize.keys():
            if labels_to_vectorize[label] == 'OneHotEncoderEstimator':
                label_indexers.append(
                    StringIndexer(inputCol=label, outputCol='label'))
            else:
                raise 'Unsupported label vectorization: ' + label

        # Chain indexers for features and labels, feature vectorizer, assembler
        pipeline = Pipeline(stages=list(map(index, CategoricalCols)) +
                            [scaling_assembler, scaler, encoder, assembler] +
                            label_indexers)

        # Train model.  This also runs the indexers.
        model = pipeline.fit(df)
        import mlflow.spark
        mlflow.spark.log_model(model, artifact_path='vectorizer')
        transformed_df = model.transform(df)
        return model, transformed_df

    @staticmethod
    def is_databricks_autotracking_enabled():
        """Check if spark.databricks.MLFlow.trackMLlib setting exists in the spark config and is enabled. 

        Returns:
            [boolean]
        """
        spark = SparkMethods.get_spark_session()
        sc = spark.sparkContext
        conf = sc.getConf().getAll()
        # check if databricks autoML tracking is enabled:
        return any(t == ('spark.databricks.mlflow.trackMLlib.enabled', 'true')
                   for t in conf)

    @staticmethod
    def grid_search_GBT(
            df,
            evaluator=MulticlassClassificationEvaluator,
            label_col='label',
            features_col='features',
            grid_params={
                'maxDepth': [3, 5, 7],
                'maxBins': [8, 16, 32],
                'maxIter': [25, 50, 100],
                'stepSize': [0.15, 0.2, 0.25]
            }):
        """Grid search for GradientBoostedTrees. This will only log all run parameters and metrics when used in databricks

        Arguments:
            df {[type]} -- [description]

        Keyword Arguments:
            label_col {str} -- [description] (default: {'label'})
            features_col {str} -- [description] (default: {'features'})
            grid_params {dict} -- [description] (default: {{'maxDepth':[3,5], 'maxBins':[16,32], 'maxIter': [25,50,100], 'stepSize':[0.15,0.2,0.25]}})

        Returns:
            [type] -- [description]
        """
        # if not SparkMethods.is_databricks_autotracking_enabled():

        # MLFlow logs
        # import mlflow
        # today = date.today()
        # experimentPath = today.strftime("%Y%m%d")
        # try:
        #     experimentID = mlflow.create_experiment(experimentPath)
        # except MlflowException:
        #     experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
        #     mlflow.set_experiment(experimentPath)
        # with mlflow.start_run(experiment_id=experimentID,run_name='GBT-training', nested=True):
        # Train a GBT model.
        gbt = GBTClassifier(featuresCol=features_col,
                            labelCol=label_col,
                            predictionCol='predicted_' + label_col)
        pipeline = Pipeline().setStages([gbt])

        # train:
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, grid_params['maxDepth'])\
            .addGrid(gbt.maxBins, grid_params['maxBins'])\
            .addGrid(gbt.maxIter, grid_params['maxIter'])\
            .addGrid(gbt.stepSize, grid_params['stepSize'])\
            .build()

        # cross validate using 80% of the CPUs
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=MulticlassClassificationEvaluator(),
            seed=24,
            parallelism=round(os.cpu_count() * 0.8),
            numFolds=3,
            collectSubModels=True)

        params_map = crossval.getEstimatorParamMaps()

        # Run cross-validation, and choose the best set of parameters.
        model = crossval.fit(df)
        transformed_df = model.bestModel.transform(df)

        bestModel = model.bestModel
        subModels = model.subModels

        return bestModel, subModels, params_map, transformed_df

    @staticmethod
    def get_MultiClassMetrics(df, label_col, model):
        transformed_df = model.transform(df)
        print('\nRESULTS FOR ' + p.upper())
        predictionAndLabels = transformed_df.select(
            F.col('predicted_' + label_col).cast(T.FloatType()),
            F.col(label_col).cast(T.FloatType())).rdd

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

    @staticmethod
    def train_GBT(df, label_col='label', features_col='features'):
        """Simple function to train GBT, to be used in conjunction with a Genetic Algorithm.

        Arguments:
            df {[type]} -- [description]

        Keyword Arguments:
            label_col {str} -- [description] (default: {'label'})
            features_col {str} -- [description] (default: {'features'})

        Returns:
            [type] -- [description]
        """
        # MLFlow logs
        # import mlflow
        # today = date.today()
        # experimentPath = today.strftime("%Y%m%d")
        # try:
        #     experimentID = mlflow.create_experiment(experimentPath)
        # except MlflowException:
        #     experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
        #     mlflow.set_experiment(experimentPath)
        # with mlflow.start_run(experiment_id=experimentID,run_name='GBT-training', nested=True):
        # Train a GBT model.
        gbt = GBTClassifier(featuresCol=features_col,
                            labelCol=label_col,
                            predictionCol='predicted_' + label_col)
        pipeline = Pipeline().setStages([gbt])

        # Run cross-validation, and choose the best set of parameters.
        model = pipeline.fit(df)
        transformed_df = model.transform(df)

        return model, transformed_df


class DataLoader:
    @staticmethod
    def load_data(data_path):
        spark = SparkMethods.get_spark_session()
        # Set Table schema and load data
        table_schema = T.StructType([
            T.StructField("age", T.IntegerType(), True),
            T.StructField("workclass", T.StringType(), True),
            T.StructField("fnlwgt", T.IntegerType(), True),
            T.StructField("education", T.StringType(), True),
            T.StructField("education-num", T.IntegerType(), True),
            T.StructField("marital-status", T.StringType(), True),
            T.StructField("occupation", T.StringType(), True),
            T.StructField("relationship", T.StringType(), True),
            T.StructField("race", T.StringType(), True),
            T.StructField("sex", T.StringType(), True),
            T.StructField("capital-gain", T.DoubleType(), True),
            T.StructField("capital-loss", T.DoubleType(), True),
            T.StructField("hours-per-week", T.DoubleType(), True),
            T.StructField("native-country", T.StringType(), True),
            T.StructField("income", T.StringType(), True)
        ])

        df = spark.read.csv(data_path,
                            ignoreLeadingWhiteSpace=True,
                            ignoreTrailingWhiteSpace=True,
                            header=False,
                            enforceSchema=False,
                            schema=table_schema)
        return df
