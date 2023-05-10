import os
import logging
import json
import numpy
import joblib
import mlflow
from mlflow.pyfunc.scoring_server import infer_and_parse_json_input
from azure.identity import ManagedIdentityCredential

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    logging.info("Init started in scoring")
    global model1, model2

    # Using os.environ to get all the environment variables
    env_vars = os.environ
    # Iterating over the dictionary to print all the environment variables
    for var in env_vars:
        print(f"{var}: {env_vars[var]}")

    credential = ManagedIdentityCredential(client_id=os.getenv("UAI_CLIENT_ID"))
    

    # More details  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models-mlflow
    mlflow.set_tracking_uri(os.getenv("TRACKING_URI"))
    logging.info("Tracking URL: {0}".format(os.getenv("TRACKING_URI")))

    model1 = mlflow.sklearn.load_model("models:/iris_svc_model/1")
    model2 = mlflow.sklearn.load_model("models:/iris_svc_model/latest")
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    json_data = json.loads(raw_data)
    if "input_data" not in json_data.keys():
        raise Exception("Request must contain a top level key named 'input_data'")
  
    
    input_data = json_data["input_data"]
    print(json.dumps(input_data))

    data = input_data["data"]
    data = numpy.array(data)
    num_rows, num_cols = data.shape # just example based on number of rows to switch model

    if ( num_rows <= 2 ) :
      logging.info(" Model 1 is predicting")
      result = model1.predict(data)
    else:
      logging.info(" Model 2 is predicting")
      result = model2.predict(data)  
    logging.info("Request processed")
    return result.tolist()
