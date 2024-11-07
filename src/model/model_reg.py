import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub
dagshub.init(repo_owner='Virupakshavegi', repo_name='MLOP-s-Water_quality_prediction_project', mlflow=True)
mlflow.set_experiment("BestandFinal_Model")
mlflow.set_tracking_uri("https://dagshub.com/Virupakshavegi/MLOP-s-Water_quality_prediction_project.mlflow")
reports_path = "reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id'] 
model_name = run_info['model_name'] 


client = MlflowClient()


model_uri = f"runs:/{run_id}/artifacts/{model_name}"


reg = mlflow.register_model(model_uri, model_name)


model_version = reg.version  


new_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")