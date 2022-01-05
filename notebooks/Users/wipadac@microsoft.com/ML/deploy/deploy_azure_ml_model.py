# Databricks notebook source
# MAGIC %pip install azureml-sdk[databricks]

# COMMAND ----------

# MAGIC %md ### Deploy latest mlFlow registry Model to Azure ML

# COMMAND ----------

# MAGIC %md ###Get Name of Model

# COMMAND ----------

dbutils.widgets.text(name = "model_name", defaultValue = "wine-model-ok", label = "Model Name")
dbutils.widgets.text(name = "stage", defaultValue = "staging", label = "Stage")
dbutils.widgets.text(name = "phase", defaultValue = "qa", label = "Phase")

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
stage = dbutils.widgets.get("stage")
phase=dbutils.widgets.get("phase")
model_name=model_name.lower()

# COMMAND ----------

# Config for AzureML Workspace
# for secure keeping, store credentials in Azure Key Vault and link using Azure Databricks secrets with dbutils
#subscription_id = dbutils.secrets.get(scope = "common-sp", key ="az-sub-id")
subscription_id = "4914f262-bda8-46cc-a9db-c9cbd694b117"
resource_group = "mlops-RG"                   
workspace_name = "mlops-AML-WS"                       
tenant_id = "72f988bf-86f1-41af-91ab-2d7cd011db47" # Tenant ID
sp_id = "293eaad2-d9fa-4d01-a718-70bccb6f8d82" # Service Principal ID
sp_secret = "yB47Q~YCeqqCpIsobltbmF0qwoHw8SAuLtUTZ" # Service Principal Secret

# COMMAND ----------

import mlflow
import mlflow.azureml
#import azureml.mlflow
import azureml
import azureml.core
from azureml.core import Workspace
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice
from random import randint

#from azureml.mlflow import get_portal_url

print("AzureML SDK version:", azureml.core.VERSION)
print("MLflow version:", mlflow.version.VERSION)

# MAGIC %md ### Get the latest version of the model that was put into Staging

# COMMAND ----------

import mlflow
import mlflow.sklearn

client = mlflow.tracking.MlflowClient()
latest_model = client.get_latest_versions(name = model_name, stages=[stage])
#print(latest_model[0])

# COMMAND ----------

model_uri="runs:/{}/model".format(latest_model[0].run_id)
latest_sk_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md ### Create or load an Azure ML Workspace

# COMMAND ----------

# MAGIC %md Before models can be deployed to Azure ML, an Azure ML Workspace must be created or obtained. The `azureml.core.Workspace.create()` function will load a workspace of a specified name or create one if it does not already exist. For more information about creating an Azure ML Workspace, see the [Azure ML Workspace management documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace).

# COMMAND ----------

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AksCompute, ComputeTarget

tenantId = "72f988bf-86f1-41af-91ab-2d7cd011db47"
clientId =  "293eaad2-d9fa-4d01-a718-70bccb6f8d82"
clientSecret = "yB47Q~YCeqqCpIsobltbmF0qwoHw8SAuLtUTZ" 

sp = ServicePrincipalAuthentication(
    tenant_id=tenantId,
    service_principal_id=clientId,
    service_principal_password=clientSecret,
)

subscription_id = "4914f262-bda8-46cc-a9db-c9cbd694b117"
resource_group = "mlops-RG"                   
workspace_name = "mlops-AML-WS"      
workspace_region = "southeastasia"
aks_compute_name = "wcpnewaks"

workspace = Workspace.get(
    name=workspace_name,
    location=workspace_region,
    resource_group=resource_group,
    subscription_id=subscription_id,
    auth=sp,
)
aks_target = AksCompute(workspace,aks_compute_name)
# COMMAND ----------

# MAGIC %md ## Building an Azure Container Image for model deployment

# COMMAND ----------

# MAGIC %md ### Use MLflow to build a Container Image for the trained model
# MAGIC 
# MAGIC We will use the `mlflow.azuereml.build_image` function to build an Azure Container Image for the trained MLflow model. This function also registers the MLflow model with a specified Azure ML workspace. The resulting image can be deployed to Azure Container Instances (ACI) or Azure Kubernetes Service (AKS) for real-time serving.

# COMMAND ----------

print(phase)

# COMMAND ----------

import mlflow.azureml

ml_name = model_name+"-"+stage

if len(ml_name)>32:
  ml_name=ml_name[0:32]

aks_deploy_config = AksWebservice.deploy_configuration(
    compute_target_name=aks_compute_name,
    cpu_cores=1,
    memory_gb=1,
    tags={"data": "wine sample", "method": "linear","alpha": str(latest_sk_model.alpha),"l1_ratio": str(latest_sk_model.l1_ratio)},
    description="Sample linear model using wine data",
)

# Note: This will create seperate service eveytime you execute, keep service_name same to update existing deployment
webservice, azure_model = mlflow.azureml.deploy(
    model_uri=model_uri,
    workspace=workspace,
    deployment_config=aks_deploy_config,
    service_name=ml_name + str(randint(10000, 99999)), 
    model_name=model_name,
    synchronous=False)                                                      

# COMMAND ----------

# MAGIC %md ### Deploy Web Service in Azure ML

# COMMAND ----------

# COMMAND ----------

webservice.wait_for_deployment(show_output=True)
webservice.scoring_uri

# COMMAND ----------

dev_scoring_uri = webservice.scoring_uri
primary, secondary = webservice.get_keys()

# COMMAND ----------

print(dev_scoring_uri)
newrest = "uri;" + dev_scoring_uri + "|"+ "key;"+ primary
# COMMAND ----------

import json
dbutils.notebook.exit(newrest)