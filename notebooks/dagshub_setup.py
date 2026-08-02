import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/akashsingh79036/model_monitoring.mlflow')

dagshub.init(repo_owner='akashsingh79036', repo_name='model_monitoring', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)