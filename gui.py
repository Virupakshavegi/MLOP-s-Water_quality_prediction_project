
import mlflow
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import threading


mlflow.set_tracking_uri("https://dagshub.com/Virupakshavegi/MLOP-s-Water_quality_prediction_project.mlflow")


model_name = "Best Model"

class PredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Water Quality Prediction")
        self.geometry("400x500")
        self.configure(bg="#eaeaea")

        self.input_frame = tk.Frame(self, bg="#ffffff", padx=20, pady=20)
        self.input_frame.pack(pady=20)

        title_label = tk.Label(self.input_frame, text="Water Quality Prediction", font=("Helvetica", 16, "bold"), bg="#ffffff")
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.create_input_fields()

        self.predict_button = tk.Button(self, text="Predict", command=self.run_prediction_thread, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.predict_button.pack(pady=20)


        self.loaded_model = self.load_model()

    def create_input_fields(self):
        self.inputs = {}
        labels = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        for idx, label in enumerate(labels):
            lbl = tk.Label(self.input_frame, text=label, bg="#ffffff", font=("Helvetica", 12))
            lbl.grid(row=idx + 1, column=0, sticky="e", pady=5)

            entry = tk.Entry(self.input_frame, width=25, font=("Helvetica", 12))
            entry.grid(row=idx + 1, column=1, pady=5)
            self.inputs[label] = entry

    def load_model(self):
        try:
         
            client = mlflow.tracking.MlflowClient()
         
            versions = client.get_latest_versions(model_name, stages=["Production"])

            if versions:
                latest_version = versions[0].version
                run_id = versions[0].run_id 

          
                logged_model = f'runs:/{run_id}/{model_name}'
             
                loaded_model = mlflow.pyfunc.load_model(logged_model)
                print(f"Model loaded from {logged_model}")
                return loaded_model
            else:
                messagebox.showerror("Error", "No model found in the 'Production' stage.")
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            return None

    def run_prediction_thread(self):
        """Run prediction in a separate thread to keep the GUI responsive."""
        thread = threading.Thread(target=self.make_prediction)
        thread.start()

    def make_prediction(self):
        try:
           
            input_data = {
                'ph': [float(self.inputs['pH'].get())],
                'Hardness': [float(self.inputs['Hardness'].get())],
                'Solids': [float(self.inputs['Solids'].get())],
                'Chloramines': [float(self.inputs['Chloramines'].get())],
                'Sulfate': [float(self.inputs['Sulfate'].get())],
                'Conductivity': [float(self.inputs['Conductivity'].get())],
                'Organic_carbon': [float(self.inputs['Organic_carbon'].get())],
                'Trihalomethanes': [float(self.inputs['Trihalomethanes'].get())],
                'Turbidity': [float(self.inputs['Turbidity'].get())]
            }

           
            data = pd.DataFrame(input_data)

            if self.loaded_model is not None:
               
                prediction = self.loaded_model.predict(data)

         
                if prediction[0] == 1:  
                    messagebox.showinfo("Prediction Result", "Water is potable.")
                else:  
                    messagebox.showinfo("Prediction Result", "Water is not potable.")
                print(f"Prediction result: {prediction[0]}") 
            else:
                messagebox.showerror("Error", "Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", "Error during prediction.")
            print(f"Error during prediction: {e}") 


if __name__ == "__main__":
    app = PredictionApp()
    app.mainloop()
