#%%
import random
from source.functions import SparkMethods, DataLoader, SparkMLBinaryClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import MultilayerPerceptronClassifier
from datetime import date
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os
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

MLP_param_options = {
    'num_hidden_layers': range(1,5),
    'first_hidden_layer_size': range(6,21,4),
    'blockSize': range(2,11,2),
    'stepSize': [0.001, 0.01, 0.1],
    'maxIter': [25, 50],
    'tol': [1e-6, 1e-4, 1e-2]
}
print(MLP_param_options)

# %%

def create_MCMC_MLP_paramMap(df, featuresCol='features', labelCol='label', max_hl_nodes=100, max_hidden_layers=3, paramMap_size=50, features_col='features', labels = ['label'], alpha=2):
    """
    Monte Carlo Markov Chain method for testing a sample of hyperparameters.
    Each hidden layer will be equal to or smaller than the prior hidden layer and chosen from the HL options.
    
    Arguments:
        df {[type]} -- must have a features and label(s) column to count the number of inputs and outputs
    
    Keyword Arguments:
        featuresCol {str} -- column with features for the NN (default: {'features'})
        labelCol {str} -- column with labels for the NN (default: {'label'})
        max_hl_nodes {int} -- Maximum number of nodes in all hidden layers (default: {100})
        max_hidden_layers {int} -- maximum number of hidden layers (default: {3})
        paramMap_size {int} -- number of hyperparameter combinations to return as a paramMap (default: {50})
        features_col {str} -- [description] (default: {'features'})
        labels {list} -- [description] (default: {['label']})
        alpha {int} -- [description] (default: {2})
    
    Returns:
        [spark MLlib paramMap] -- Same structure as paramGridBuilder().addGrid(...).build()
    """

    Ni = df.select(featuresCol).head()[featuresCol].size # number of input neurons
    No = df.select(labelCol).distinct().count() # number of output neurons
    #Ns = train_df.count() # number of samples in training data
    
    #Nh = round(Ns / (alpha*(Ni+No))) # max number of nodes based on heuristics, does not work on large datasets, should never be reached but just in case

    def get_random_MLP_options(input_size, output_size, sample_size, max_nodes, MLP_param_options):
        """Monte Carlo Markov Chain method for testing a sample of hyperparameters.
        Each hidden layer will be equal to or smaller than the prior hidden layer and chosen from the HL options.
        
        Arguments:
            input_size {[type]} -- [description]
            output_size {[type]} -- [description]
            sample_size {[type]} -- [description]
            max_nodes {[type]} -- [description]
            MLP_param_options {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        MCMC_grid = []

        while len(MCMC_grid) < sample_size:
            MLP_params = {}

            # choose random options for all parameters
            for key in MLP_param_options.keys():
                MLP_params[key] = random.choice(MLP_param_options[key])

            # set size of first hidden layer
            MLP_params['hidden_layers'] = [MLP_params['first_hidden_layer_size']]
            del MLP_params['first_hidden_layer_size']

            #select equal or smaller hidden layer size based on previous hidden layer
            for l in range(2,MLP_params['num_hidden_layers']+1):
                options_left = [x for x in MLP_param_options['first_hidden_layer_size'] if x <= min(MLP_params['hidden_layers'])]
                MLP_params['hidden_layers'].append(random.choice(options_left))

            if (sum(MLP_params['hidden_layers'])< max_nodes):
                MLP_params['layers'] = [input_size] + MLP_params['hidden_layers'] + [output_size]
                del MLP_params['hidden_layers']
                del MLP_params['num_hidden_layers']
                MCMC_grid.append(MLP_params)

        return MCMC_grid

    NN_options = get_random_MLP_options(Ni, No, paramMap_size, max_hl_nodes, MLP_param_options)

    paramGrid = SparkMethods.build_param_grid(MultilayerPerceptronClassifier(), NN_options)
    return paramGrid

print(create_MCMC_MLP_paramMap(trainingData))


#%%

def grid_search_MLP(
        train_df, test_df,
        evaluator='MulticlassClassificationEvaluator',
        label_col='label',
        features_col='features',
        model_file_name='bestMLP',
        kfolds=5,
        grid_params={
            'blockSize': [10],
            'stepSize': [0.1],
            'layers': [[123, 18, 14,10,2]],
            'maxIter': [25],
            'tol': [1e-6, 1e-4]
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
    with mlflow.start_run(experiment_id=experimentID,run_name='MLP-training', nested=True):
        # Train a GBT model.
        MLP = MultilayerPerceptronClassifier(
                            predictionCol='predicted_' + label_col)
        
        
        pipeline = Pipeline().setStages([MLP])

        # train:
        paramGrid = ParamGridBuilder() \
            .addGrid(MLP.blockSize, grid_params['blockSize'])\
            .addGrid(MLP.stepSize, grid_params['stepSize'])\
            .addGrid(MLP.layers, grid_params['layers'])\
            .addGrid(MLP.maxIter, grid_params['maxIter'])\
            .addGrid(MLP.tol, grid_params['tol'])\
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

#%%
models = grid_search_MLP(trainingData, testData)
#%%

# TBD - create all combinations and then randomly choose from list and pop to avoid testing the same set of parameters twice

#%%
models.bestModel.stages[0].weights

# %%
