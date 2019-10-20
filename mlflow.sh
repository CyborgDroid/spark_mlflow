#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spark_mlflow
mlflow ui --port=5148