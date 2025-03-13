# creating custom environment

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

# workspace connecction
ml_client = MLClient.from_config(credential=DefaultAzureCredential())


# new environment configuration
new_env = Environment(
    name = "custom_env",
    description = "custom environment for loan approval model",
    conda_file = "custom_environment.yaml",
    image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest" 
)

# creating new environment
ml_client.environments.create_or_update(new_env)
print(f"new environment created and registered to the workspace")
