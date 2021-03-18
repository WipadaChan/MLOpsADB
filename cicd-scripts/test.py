import importlib,pprint,json,os, requests
from  mlflow_http_client import MlflowHttpClient, get_host,get_token
os.environ["DATABRICKS_HOST"] = "https://adb-6618238979725084.4.azuredatabricks.net"
os.environ["DATABRICKS_TOKEN"] = "dapibb5095c3e194d0131da6ac8d843d3941"
#access_token = 'dapibb5095c3e194d0131da6ac8d843d3941'

token=get_token()
print(token)
#headers={'Authorization': 'Bearer {}'.format(access_token)}
#uri = "https://adb-6618238979725084.4.azuredatabricks.net/api/2.0/mlflow/registered-models/get-latest-versions?name=tech-summit-wine-model&stages=staging"
#rsp = requests.get(uri, headers=headers)

#print(json.loads(rsp.text))