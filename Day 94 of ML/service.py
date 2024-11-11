import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from train import *  

# Load the BentoML runner for the saved model
iris_clf_runner = bentoml.sklearn.get("iris_classifier:gsjmt5fapwmvq4fg").to_runner()

# Define the BentoML service
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# Define the API endpoint
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    # Use the runner to make predictions
    result = iris_clf_runner.predict.run(input_series)
    return result
