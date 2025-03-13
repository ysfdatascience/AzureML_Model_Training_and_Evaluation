from dotenv import load_dotenv
import os


load_dotenv()

sub_id = os.getenv("sub_id")
rg_name = os.getenv("rg_name")
ws_name = os.getenv("ws_name")



from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute



ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=sub_id,
    resource_group_name=rg_name,
    workspace_name=ws_name
)

list(ml_client.workspaces.list())


# Creating Compute Cluster
new_cc = AmlCompute(
    name = "yy-compute-cluster",
    size = "Standard_DS3_v2",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=300,
    tier="Dedicated",
    tags={
        "purpose": "AML"
    }
)

ml_client.compute.begin_create_or_update(new_cc).wait()
print("yeni compute cluster oluşturuldu")


# Creating Compute Instance
from azure.ai.ml.entities import ComputeInstance


new_ci = ComputeInstance(
    name = "yy-compute-instance",
    size="Standard_DS3_v2",
    location = "eastus2",
    description="this compute instance is for developing loan approval model",
    tags={
        "purpose": "loan model development"
    },
    idle_time_before_shutdown_minutes=15,
    
)

ml_client.compute.begin_create_or_update(new_ci).wait()
print("yeni compute instance oluşturuldu")
