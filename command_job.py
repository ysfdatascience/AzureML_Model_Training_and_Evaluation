# creating a command job for the script

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

# creating workspace connection
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

env = ml_client.environments.get(name = "custom_env", version = 1).id
data_path = ml_client.data.get(name = "loan_Data", version = 1).path

#creating command job
job = command(
    name = "loan_model_training_1",
    experiment_name= "loan_model_training",
    environment=env,
    code = ".",
    inputs = dict(
        input_data = Input(type = AssetTypes.URI_FILE, path = data_path),
        n_estimators = 100,
        min_sample_split = 20,
        test_ratio = 0.20,
        random_state = 365 
    ),

    command = """python train_and_eval.py --input_data ${{inputs.input_data}} 
                --n_estimators ${{inputs.n_estimators}} --min_sample_split ${{inputs.min_sample_split}} 
                --test_ratio ${{inputs.test_ratio}} --random_state ${{inputs.random_state}}""",

    compute = "yy-compute-instance"
)

ml_client.jobs.create_or_update(job)
