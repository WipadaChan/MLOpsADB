# Introduction 

#THIS CODE IS PROVIDED WITH ABSOLUTELY NO WARRANTY OR SUPPORT AND SHOULD ONLY SERVE AS A DEMO OF THE AVAILABLE FEATURES

## Prerequisit 
- Azure ML Workspace
- Azure Databricks workspace
- Create Service Principle 
- Grant Permission to SP in Azure ML workspace + resource group (role:Contributor)
- Create Variable for Pipeline in Azure DevOps:
    - DATABRICKS_HOST
    - DATABRICKS_TOKEN
    - EXISTING_CLUSTER_ID
    - MODEL_NAME
To update Azure ML Workspace Config:
    - Edit in file deploy_azure_ml_model.py

This project demonstrates the use of Azure DevOps as the tool to work with ML Pipelines in Azure Databricks.

Workflow Steps

- Data Scientist promotes model into Staging (e.g. as in the train_wine_model Notebook): Train model on Databrick workspace and register model to MLflow (train_model.py)
- Pipeline gets triggered (via Webhooks, manually or via API, as in the train_wine_model Notebook)
- Azure Dev Ops uploads deployment notebook (deploy_azure_ml_model) from git to a dedicated Test/QA region within the workspace via the Databricks workspace API
- Azure Dev Ops runs deploy notebook with creater job and run submit which does the following:
    - Retrieves latest model staging from registry
    - Deploys model as an Azure ML model and creates an image
    - Deploys REST API for he model/image
    - returns an the REST API deployment URL to Azure Dev Ops
- Azure Dev Ops uploads test notebook from git to a dedicated Test/QA region within the workspace
- Azure Dev Ops runs deploy notebook(same deploy notebook with different parameter) with run submit which does the following:
    - Retrieves test data
    - Invokes REST API
- If successful, DevOps will deploy the model into production using the mlFlow REST API# databricksMLOpsAzureDemo
