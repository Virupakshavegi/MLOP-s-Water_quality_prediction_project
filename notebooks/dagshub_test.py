import dagshub
import mlflow


dagshub.init(repo_owner='Virupakshavegi', repo_name='MLOP-s-Water_quality_prediction_project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Virupakshavegi/MLOP-s-Water_quality_prediction_project.mlflow")

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)