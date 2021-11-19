# Flask app

### Set up local dev environment:

```python -m venv ~/.dl_ve```

```source ~/.dl_ve/bin/activate```

```pip install -r requirements.txt```

```python main.py```


App available at: ```http://0.0.0.0:80/```


### Deploy to Google App Engine

1. Select project and enable cloudapi from gcloud shell

	```gcloud services enable cloudbuild.googleapis.com```

2. Create a project from cloud editor and upload the project files

3. Deploy from root of project

	```gcloud app deploy```

4. Access web app here:
	https://alzheimers-331518.uc.r.appspot.com

