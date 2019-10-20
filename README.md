# Overview #

Demonstration of how to use MLFlow to:
1. Track a grid search using spark MLlib Gradient Boosted Trees, Linear Regressions, and hyperparameter tuning of these.
2. Predict with a saved model via a script
3. Create a microservice of the model with a dockerized API endpoint.

### Configure conda/miniconda ###

If you haven't already done so, add conda-forge as a channel
```
conda config --add channels conda-forge
conda config --set channel_priority false
```
Create the environment:
```
conda create --name spark_mlflow python=3.7 jupyter pandas-profiling pyspark pyarrow mlflow unittest
```

### Download the sample data: ###

```
python source/get_data.py 
```
## MLFlow UI ###

### To run ad-hoc, execute the following in the terminal within the folder of the project: ###
```
conda activate spark_mlflow
mlflow ui
```
### To start MLFlow UI in the background instead to run indefinitely: ###
_Assumes NodeJS and npm are installed_
http://pm2.keymetrics.io/docs/usage/quick-start/

1. Install PM2 (process manager)
    ```
    npm install pm2@latest -g
    # or 
    yarn global add pm2
    ```
2. Start mlflow:
    ```
    pm2 start mlflow.sh
    # check logs for succesfull initialization:
    pm2 logs mlflow.sh
    ```
    mlflow.sh must have execution rights, if the above 
3. If you want MLFlow UI to start on every reboot:
    ```
    pm2 save
    pm2 startup
    ```