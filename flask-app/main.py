import os
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request, redirect
import json
import nsvision as nv
import requests
import googleapiclient.discovery
import numpy as np


#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/princy_joy/flask-app/secret/alzheimers-331518-fca7b6bc902a.json"
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
    return render_template('home.html')


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST' and request.files['file'].filename != '':
      
      f = request.files['file'].read()
      f_open = open('/tmp/img1', 'wb')
      f_open.write(f)
      image = nv.imread('/tmp/img1',resize=(50,50), color_mode='rgb')
      image = nv.expand_dims(image,axis=0)
      data = ""
      try:
         classes ={'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented' :3}
         idc = {k:v for v, k in classes.items()}
         
         response = predict_json("alzheimers-331518", "kubeflow_pipelines_alzheimers", image.tolist())
         
         output = response[0]
         data = idc[np.argmax(output)] 
         
         os.remove('/tmp/img1')
      except Exception as e: 
         print(e)
         print("error")
         os.remove('/tmp/img1')
         return render_template('uploader.html')

      return render_template('uploader.html', value=data)
   else:
      return render_template('uploader.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
