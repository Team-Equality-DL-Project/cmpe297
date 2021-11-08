# Set up
### Step 1: Create a cluster (make sure you delete the cluster when done)
1. Go to https://console.cloud.google.com/marketplace/kubernetes/config/google-cloud-ai-platform/kubeflow-pipelines?version=1.7&project=alzheimers-331518&authuser=1
2. Create the cluster and deploy. Set zone to "us-west1-b" and name the pipeline "kubeflow-pipelines-alzheimers"
    ![s](images/kubeflow_cluster_setup.png)
### Step 2: Start the Jupyter notebook
1. Go to https://console.cloud.google.com/vertex-ai/workbench/list/instances?authuser=1&project=alzheimers-331518
2. Start the notebook
### Step 3: Clone the code
Create a new notebook to run some code to pull from our Githbu repo. Get your Github access token with these steps and fill in your Github username and personal access token

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

```
username = "<your-github-username>"
token = "<your-github-access-token>"

!git clone https://{username}:{token}@github.com/Team-Equality-DL-Project/cmpe297.git /home/jupyter/cmpe297
```
### Step 4:
To deploy the latest pipeline, run `cmpe297/kubeflow/main.ipynb`