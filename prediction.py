import mlflow
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/Virupakshavegi/MLOP-s-Water_quality_prediction_project.mlflow")


model_name = "Best Model"  

try:
   
    client = mlflow.tracking.MlflowClient()

    
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if versions:
        latest_version = versions[0].version
        run_id = versions[0].run_id  
        print(f"Latest version in Production: {latest_version}, Run ID: {run_id}")

      
        logged_model = f'runs:/{run_id}/{model_name}'
        print("Logged Model:", logged_model)

       
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print(f"Model loaded from {logged_model}")

     
        data = pd.DataFrame({
            'ph': [3.71608],
            'Hardness': [204.89045],
            'Solids': [20791.318981],
            'Chloramines': [7.300212],
            'Sulfate': [368.516441],
            'Conductivity': [564.308654],
            'Organic_carbon': [10.379783],
            'Trihalomethanes': [86.99097],
            'Turbidity': [2.963135]
        })

   
        prediction = loaded_model.predict(data)
        print("Prediction:", prediction)
    else:
        print("No model found in the 'Production' stage.")

except Exception as e:
    print(f"Error fetching model: {e}")