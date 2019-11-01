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
        Can also be used to ensure other stratifications are used like gender, age groups, or cluster. 

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
                   MinMaxCols=[],
                   vectorizer_file_name='vectorizer'):
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
        experimentPath = date.today().strftime("%Y%m%d")
        import mlflow
        print(experimentPath)
        try:
            print('created new MLFlow Experiment')
            experimentID = mlflow.create_experiment(experimentPath)
        except MlflowException:
            print('Using existing MLFlow Experiment')
            experimentID = MlflowClient().get_experiment_by_name(
                experimentPath).experiment_id
        mlflow.set_experiment(experimentPath)

        # with mlflow.start_run(experiment_id=experimentID,run_name='vectorizer', nested=True):
        params = {'CategoricalCols': CategoricalCols, 'MinMaxCols': MinMaxCols}
        params.update(labels_to_vectorize)
        mlflow.set_tag('vectorizer', vectorizer_file_name)
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
                            [scaling_assembler, scaler, encoder, assembler])

        label_pipeline = Pipeline(stages=label_indexers)
        label_vectorizer = label_pipeline.fit(df)

        # Train model.  This also runs the indexers.
        vectorizer = pipeline.fit(df)
        import mlflow.spark
        mlflow.spark.log_model(vectorizer, artifact_path=vectorizer_file_name)
        mlflow.spark.log_model(label_vectorizer, artifact_path='label_' + vectorizer_file_name) 
        transformed_df = vectorizer.transform(df)
        #label vectorizer is not saved to the regular vectorizer pipeline since new data will not be labelled
        transformed_df = label_vectorizer.transform(transformed_df)
        return vectorizer, label_vectorizer, transformed_df

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
    def get_model_params(model):
        """Get params for a spark model (bestModel or subModels)
        
        Arguments:
            model {spark model}
        
        Returns:
            [dict] -- List of dicts with the params for all stages in the model
        """
        # get names for all stages
        stage_names = [s.uid for s in model.stages]
        # get params for all stages
        all_params = []
        for s in model.stages:
            p = {
                'stage' : s.uid
            }
            raw_params = s.extractParamMap()
            p.update({p.name:raw_params.get(p) for p in raw_params})
            all_params.append(p)
        return all_params

    @staticmethod
    def get_MultiClassMetrics(df, model=None, data_type='train', label_col='label'):
        """Get multiclass metrics (this needs to be distributed with multiprocessing to speed up grid search)
        
        Arguments:
            df {pyspark.sql.dataframe.DataFrame}
            model {pyspark.ml.pipeline.PipelineModel} 
        
        Keyword Arguments:
            data_type {str} -- train/test/val - will be appended to metric keys (default: {'train'})
            label_col {str} -- Column name for the actual label (default: {'label'})
        
        Returns:
            {dict} -- Dictionary of model metrics ready to log to MLFlow or print
        """
        # if the DF has not been transformed by the model, apply model.
        if model!=None and 'predicted_'+label_col not in df.columns:
            df = model.transform(df)

        predictionAndLabels = df.select(
            F.col('predicted_' + label_col).cast(T.FloatType()),
            F.col(label_col).cast(T.FloatType())).rdd

        metrics = MulticlassMetrics(predictionAndLabels)

        # Overall statistics
        log_metrics = {
            data_type + 'Precision' : metrics.precision(),
            data_type + 'Recall' : metrics.recall(),
            data_type + 'F1Score' : metrics.fMeasure(),
            data_type + 'WeightedRecall': metrics.weightedRecall,
            data_type + 'WeightedPrecision': metrics.weightedPrecision,
            data_type + 'WeightedF1Score': metrics.weightedFMeasure(),
            data_type + 'WeightedF0.5Score': metrics.weightedFMeasure(beta=0.5),
            data_type + 'WeightedFalsePositiveRate': metrics.weightedFalsePositiveRate 
        }

        # get labels (ex: 0.0 and 1.0)
        labels = [x[label_col] for x in df.select(label_col).distinct().collect()]
        for label in sorted(labels):
            log_metrics.update({data_type + 'Recall' + str(label): float(metrics.recall(label))})
            log_metrics.update({data_type + 'Precision' + str(label):float(metrics.precision(label))})
            log_metrics.update({data_type + 'F1Score' + str(label): float(metrics.fMeasure(label, beta=1.0))})

        return log_metrics


    @staticmethod
    def combine_vectorizer_and_model(vectorizer, model, model_file_name,log_model=True):
        combined_model = Pipeline(stages=[vectorizer, model])
        if log_model:
            import mlflow.spark
            mlflow.spark.log_model(combined_model, 'vectorizer_and_' + model_file_name)
        return 

class SparkMLBinaryClassifier:
    """[summary]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self,
            train_df, test_df,
            evaluator='MulticlassClassificationEvaluator',
            label_col='label',
            features_col='features',
            kfolds=5,
            GBT_params={
                'maxDepth': [3, 5, 7],
                'maxBins': [8, 16, 32],
                'maxIter': [25, 50, 100],
                'stepSize': [0.15, 0.2, 0.25],
            
            },
            LSVC_params={
                'standardization': [True, False],
                'aggregationDepth': [2, 5, 7],
                'regParam': [0.1, 1, 10],
                'maxIter': [25, 50, 100],
                'tol': [1e-6, 1e-4, 1e-2]
            },
            MLP_params = {
                'max_hidden_layers': 3,
                'blockSize': [2, 5, 10],
                'stepSize': [0.001, 0.01],
                'maxIter': [25],
                'tol': [1e-6, 1e-4]
            }):
        general_params = {
           'train_df': train_df, 
           'test_df': test_df,
            'evaluator':evaluator,
            'label_col':label_col,
            'features_col':features_col,
            'kfolds':kfolds 
        }
        self.models = {}
        self.models['GBT'] = SparkMLBinaryClassifier.grid_search_GBT(**general_params, grid_params=GBT_params)
        self.models['LSVC'] = SparkMLBinaryClassifier.grid_search_LSVC(**general_params, grid_params=LSVC_params)


    @staticmethod
    def grid_search_GBT(
            train_df, test_df,
            evaluator='MulticlassClassificationEvaluator',
            label_col='label',
            features_col='features',
            model_file_name='bestGBT',
            kfolds=5,
            grid_params={
                'maxDepth': [3, 5, 7],
                'maxBins': [8, 16, 32],
                'maxIter': [25, 50, 100],
                'stepSize': [0.15, 0.2, 0.25]
            }):
        """Grid search for GradientBoostedTrees.
        
        Arguments:
            train_df {[type]} -- [description]
            test_df {[type]} -- [description]
        
        Keyword Arguments:
            evaluator {str} -- [To be added to enable BinaryClassificationEvaluator and metrics] (default: {'MulticlassClassificationEvaluator'})
            label_col {str} -- [The target label column] (default: {'label'})
            features_col {str} -- [column with sparse matrix features] (default: {'features'})
            model_file_name {str} -- [filename for the model] (default: {'bestGBT'})
            grid_params {dict} -- [description] (default: {{'maxDepth': [3, 5, 7],'maxBins': [8, 16, 32],'maxIter': [25, 50, 100],'stepSize': [0.15, 0.2, 0.25]}})
        
        Returns:
            [type] -- [description]
        """
        # if not SparkMethods.is_databricks_autotracking_enabled():

        # MLFlow logs
        import mlflow
        today = date.today()
        experimentPath = today.strftime("%Y%m%d")
        try:
            print('created new MLFlow Experiment GBT') 
            experimentID = mlflow.create_experiment(experimentPath)
        except MlflowException:
            print('Using existing MLFlow Experiment GBT') 
            experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
        
        # start nested mlflow experiement
        with mlflow.start_run(experiment_id=experimentID,run_name='GBT-training', nested=True):
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
                evaluator=MulticlassClassificationEvaluator(predictionCol='predicted_' + label_col, labelCol=label_col),
                seed=1,
                parallelism=round(os.cpu_count() * 0.8),
                numFolds=kfolds,
                collectSubModels=True)

            params_map = crossval.getEstimatorParamMaps()

            # Run cross-validation, and choose the best set of parameters.
            cv_model = crossval.fit(train_df)

            # Log Parameters, Metrics, and all models with MLFlow

            bestModel = cv_model.bestModel
            subModels = cv_model.subModels

            #log params and metrics for all runs with MLFlow
            for i, fold in enumerate(subModels):
                # print('\n***Fold ' + str(i + 1) + '***\n')
                for x, model in enumerate(fold):
                    # print('\nModel ' + str(x + 1))
                    train_metrics = SparkMethods.get_MultiClassMetrics(train_df, model, data_type='train', label_col=label_col)
                    test_metrics = SparkMethods.get_MultiClassMetrics(test_df, model, data_type='test', label_col=label_col)
                    with mlflow.start_run(experiment_id=experimentID,run_name='GBT-training', nested=True):
                        mlflow.log_param('Model', x + 1)
                        mlflow.log_param('current_Kfold', i + 1)
                        mlflow.log_param('numKfolds', kfolds)
                        model_stages_params = SparkMethods.get_model_params(model)
                        for stage_params in model_stages_params:
                            mlflow.log_params(stage_params)
                        mlflow.log_metrics(train_metrics)
                        mlflow.log_metrics(test_metrics)

        # log best model, params, and metrics with MLFlow
        print('here')
        params_stages_bestModel = SparkMethods.get_model_params(bestModel)
        train_metrics_bestModel = SparkMethods.get_MultiClassMetrics(train_df, bestModel, data_type='train', label_col=label_col)
        test_metrics_bestModel = SparkMethods.get_MultiClassMetrics(test_df, bestModel, data_type='test', label_col=label_col)
        for params_stages in params_stages_bestModel:
            mlflow.log_params(params_stages)
        mlflow.set_tag('model', model_file_name)
        mlflow.log_metrics(train_metrics_bestModel)
        mlflow.log_metrics(test_metrics_bestModel)
        import mlflow.spark
        mlflow.spark.log_model(bestModel, model_file_name)
        train_df = cv_model.bestModel.transform(train_df)
        test_df = cv_model.bestModel.transform(test_df)
        mlflow.end_run()

        return cv_model

    @staticmethod
    def grid_search_LSVC(
            train_df, test_df,
            evaluator='MulticlassClassificationEvaluator',
            label_col='label',
            features_col='features',
            model_file_name='bestLSVC',
            kfolds=5,
            grid_params={
                'standardization': [True, False],
                'aggregationDepth': [2, 5, 7],
                'regParam': [0.1, 1, 10],
                'maxIter': [25, 50, 100],
                'tol': [1e-6, 1e-4, 1e-2]
            }):
        """Grid search for Linear SVMs/SVCs.
        
        Arguments:
            train_df {[type]} -- [description]
            test_df {[type]} -- [description]
        
        Keyword Arguments:
            evaluator {str} -- [To be added to enable BinaryClassificationEvaluator and metrics] (default: {'MulticlassClassificationEvaluator'})
            label_col {str} -- [The target label column] (default: {'label'})
            features_col {str} -- [column with sparse matrix features] (default: {'features'})
            model_file_name {str} -- [filename for the model] (default: {'bestGBT'})
            grid_params {dict} -- [description] (default: {{'maxDepth': [3, 5, 7],'maxBins': [8, 16, 32],'maxIter': [25, 50, 100],'stepSize': [0.15, 0.2, 0.25]}})
        
        Returns:
            [type] -- [description]
        """
        # if not SparkMethods.is_databricks_autotracking_enabled():

        # MLFlow logs
        import mlflow
        today = date.today()
        experimentPath = today.strftime("%Y%m%d")
        try:
            print('created new MLFlow Experiment LSVC') 
            experimentID = mlflow.create_experiment(experimentPath)
        except MlflowException:
            print('Using existing MLFlow Experiment LSVC') 
            experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
        
        # start nested mlflow experiement
        with mlflow.start_run(experiment_id=experimentID,run_name='LSVC-training', nested=True):
            # Train a GBT model.
            LSVC = LinearSVC(featuresCol=features_col,
                                labelCol=label_col,
                                predictionCol='predicted_' + label_col)
            
            
            pipeline = Pipeline().setStages([LSVC])

            # train:
            paramGrid = ParamGridBuilder() \
                .addGrid(LSVC.standardization, grid_params['standardization'])\
                .addGrid(LSVC.aggregationDepth, grid_params['aggregationDepth'])\
                .addGrid(LSVC.regParam, grid_params['regParam'])\
                .addGrid(LSVC.maxIter, grid_params['maxIter'])\
                .addGrid(LSVC.tol, grid_params['tol'])\
                .build()

            # cross validate using 80% of the CPUs
            crossval = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=MulticlassClassificationEvaluator(predictionCol='predicted_' + label_col, labelCol=label_col),
                seed=1,
                parallelism=round(os.cpu_count() * 0.8),
                numFolds=kfolds,
                collectSubModels=True)

            params_map = crossval.getEstimatorParamMaps()

            # Run cross-validation, and choose the best set of parameters.
            cv_model = crossval.fit(train_df)

            # Log Parameters, Metrics, and all models with MLFlow

            bestModel = cv_model.bestModel
            subModels = cv_model.subModels

            #log params and metrics for all runs with MLFlow
            for i, fold in enumerate(subModels):
                # print('\n***Fold ' + str(i + 1) + '***\n')
                for x, model in enumerate(fold):
                    # print('\nModel ' + str(x + 1))
                    train_metrics = SparkMethods.get_MultiClassMetrics(train_df, model, data_type='train', label_col=label_col)
                    test_metrics = SparkMethods.get_MultiClassMetrics(test_df, model, data_type='test', label_col=label_col)
                    with mlflow.start_run(experiment_id=experimentID,run_name='LSVC-training', nested=True):
                        mlflow.log_param('Model', x + 1)
                        mlflow.log_param('current_Kfold', i + 1)
                        mlflow.log_param('numKfolds', kfolds)
                        model_stages_params = SparkMethods.get_model_params(model)
                        for stage_params in model_stages_params:
                            mlflow.log_params(stage_params)
                        mlflow.log_metrics(train_metrics)
                        mlflow.log_metrics(test_metrics)

        # log best model, params, and metrics with MLFlow
        print('here')
        params_stages_bestModel = SparkMethods.get_model_params(bestModel)
        train_metrics_bestModel = SparkMethods.get_MultiClassMetrics(train_df, bestModel, data_type='train', label_col=label_col)
        test_metrics_bestModel = SparkMethods.get_MultiClassMetrics(test_df, bestModel, data_type='test', label_col=label_col)
        for params_stages in params_stages_bestModel:
            mlflow.log_params(params_stages)
        mlflow.set_tag('model', model_file_name)
        mlflow.log_metrics(train_metrics_bestModel)
        mlflow.log_metrics(test_metrics_bestModel)
        import mlflow.spark
        mlflow.spark.log_model(bestModel, model_file_name)
        train_df = cv_model.bestModel.transform(train_df)
        test_df = cv_model.bestModel.transform(test_df)
        mlflow.end_run()

        return cv_model

    @staticmethod
    def grid_search_MLP(
            train_df, test_df,
            evaluator='MulticlassClassificationEvaluator',
            label_col='label',
            features_col='features',
            model_file_name='bestMLP',
            kfolds=5,
            grid_params = {
                'max_hidden_layers': 3,
                'blockSize': [2, 5, 10],
                'stepSize': [0.001, 0.01],
                'maxIter': [25],
                'tol': [1e-6, 1e-4]
            }):
        """Grid search for MLP with randomized hidden layer complexity
        
        Arguments:
            train_df {[type]} -- [description]
            test_df {[type]} -- [description]
        
        Keyword Arguments:
            evaluator {str} -- [To be added to enable BinaryClassificationEvaluator and metrics] (default: {'MulticlassClassificationEvaluator'})
            label_col {str} -- [The target label column] (default: {'label'})
            features_col {str} -- [column with sparse matrix features] (default: {'features'})
            model_file_name {str} -- [filename for the model] (default: {'bestGBT'})
            grid_params {dict} -- [description] (default: {{'maxDepth': [3, 5, 7],'maxBins': [8, 16, 32],'maxIter': [25, 50, 100],'stepSize': [0.15, 0.2, 0.25]}})
        
        Returns:
            [type] -- [description]
        """
        # if not SparkMethods.is_databricks_autotracking_enabled():

        # MLFlow logs
        import mlflow
        today = date.today()
        experimentPath = today.strftime("%Y%m%d")
        try:
            print('created new MLFlow Experiment LSVC') 
            experimentID = mlflow.create_experiment(experimentPath)
        except MlflowException:
            print('Using existing MLFlow Experiment LSVC') 
            experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
        
        # start nested mlflow experiement
        with mlflow.start_run(experiment_id=experimentID,run_name='LSVC-training', nested=True):
            # Train a GBT model.
            LSVC = LinearSVC(featuresCol=features_col,
                                labelCol=label_col,
                                predictionCol='predicted_' + label_col)
            
            
            pipeline = Pipeline().setStages([LSVC])

            # train:
            paramGrid = ParamGridBuilder() \
                .addGrid(LSVC.standardization, grid_params['standardization'])\
                .addGrid(LSVC.aggregationDepth, grid_params['aggregationDepth'])\
                .addGrid(LSVC.regParam, grid_params['regParam'])\
                .addGrid(LSVC.maxIter, grid_params['maxIter'])\
                .addGrid(LSVC.tol, grid_params['tol'])\
                .build()

            # cross validate using 80% of the CPUs
            crossval = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=MulticlassClassificationEvaluator(predictionCol='predicted_' + label_col, labelCol=label_col),
                seed=1,
                parallelism=round(os.cpu_count() * 0.8),
                numFolds=kfolds,
                collectSubModels=True)

            params_map = crossval.getEstimatorParamMaps()

            # Run cross-validation, and choose the best set of parameters.
            cv_model = crossval.fit(train_df)

            # Log Parameters, Metrics, and all models with MLFlow

            bestModel = cv_model.bestModel
            subModels = cv_model.subModels

            #log params and metrics for all runs with MLFlow
            for i, fold in enumerate(subModels):
                # print('\n***Fold ' + str(i + 1) + '***\n')
                for x, model in enumerate(fold):
                    # print('\nModel ' + str(x + 1))
                    train_metrics = SparkMethods.get_MultiClassMetrics(train_df, model, data_type='train', label_col=label_col)
                    test_metrics = SparkMethods.get_MultiClassMetrics(test_df, model, data_type='test', label_col=label_col)
                    with mlflow.start_run(experiment_id=experimentID,run_name='LSVC-training', nested=True):
                        mlflow.log_param('Model', x + 1)
                        mlflow.log_param('current_Kfold', i + 1)
                        mlflow.log_param('numKfolds', kfolds)
                        model_stages_params = SparkMethods.get_model_params(model)
                        for stage_params in model_stages_params:
                            mlflow.log_params(stage_params)
                        mlflow.log_metrics(train_metrics)
                        mlflow.log_metrics(test_metrics)

        # log best model, params, and metrics with MLFlow
        print('here')
        params_stages_bestModel = SparkMethods.get_model_params(bestModel)
        train_metrics_bestModel = SparkMethods.get_MultiClassMetrics(train_df, bestModel, data_type='train', label_col=label_col)
        test_metrics_bestModel = SparkMethods.get_MultiClassMetrics(test_df, bestModel, data_type='test', label_col=label_col)
        for params_stages in params_stages_bestModel:
            mlflow.log_params(params_stages)
        mlflow.set_tag('model', model_file_name)
        mlflow.log_metrics(train_metrics_bestModel)
        mlflow.log_metrics(test_metrics_bestModel)
        import mlflow.spark
        mlflow.spark.log_model(bestModel, model_file_name)
        train_df = cv_model.bestModel.transform(train_df)
        test_df = cv_model.bestModel.transform(test_df)
        mlflow.end_run()

        return cv_model

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
