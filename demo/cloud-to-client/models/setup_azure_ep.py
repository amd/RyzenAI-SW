import argparse
import json
import random
import yaml
import tempfile
import time
from pathlib import Path
from shutil import copy
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import ManagedOnlineDeployment, Model
from azure.core.exceptions import ResourceNotFoundError


def parse_args():
    parser = argparse.ArgumentParser('azure_ep')
    parser.add_argument('--prefix', type=str, required = True, help = 'config.json file for your azure workspace')
    parser.add_argument('--config','-c', type=str, default='config.json', help = 'config.json file for your azure workspace')
    parser.add_argument('--model','-m', type=str, choices = ['yolov5', 'retinaface'], default='retinaface', help = 'model name')
    parser.add_argument('--step','-s', type=str, choices = ['endpoint', 'deployment', 'set_traffic'], help = 'model name')
    parser.add_argument('--instance-type', type=str, default='Standard_F2s_v2', help = 'Instance SKU type')
    parser.add_argument('--instance-count', type=int, default=1, help = 'Instance count')
    args = parser.parse_args()
    return args


def get_client(config):
    p = Path(config)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with open(p) as f:
        azure_config = json.load(f)

        ml_client = MLClient(
            InteractiveBrowserCredential(),
            # DefaultAzureCredential(),
            azure_config['subscription_id'],
            azure_config['resource_group'],
            azure_config['workspace_name'],
        )
    return ml_client

def get_endpoint_name(model):
    global EP_PREFIX
    return f'{EP_PREFIX}-{model}'

def get_endpoint(ml_client, model):
    endpoint_name = get_endpoint_name(model)
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    except:
        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
        ml_client.online_endpoints.begin_create_or_update(endpoint)
    return endpoint


def set_traffic(ml_client, model):
    endpoint_name, endpoint = get_endpoint(ml_client, model)
    endpoint.traffic = {model: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint)


def prepare_model_path(p, az_model_name, az_model_version):
    model_file = list((p / 'model').glob('*.onnx'))
    assert len(model_file) == 1, f"Multiple onnx models have been found under {str(p/'model')}"
    tmp_dir = tempfile.TemporaryDirectory()
    src = model_file[0]
    dst = Path(tmp_dir.name) / az_model_name / az_model_version / 'model.onnx'
    dst.parent.mkdir(parents = True, exist_ok = True)
    copy(src, dst)
    return tmp_dir


def deploy(ml_client, args):
    global EP_PREFIX
    model, instance_type, instance_count = args.model, args.instance_type, args.instance_count
    p = Path(__file__).parent.resolve() / model
    endpoint_name = get_endpoint_name(model)
    if not p.is_dir():
        raise FileNotFoundError(f'Cannot find model directory: {str(p)}')
    with open(p / 'azure_config.yaml.example') as f:
        az_config = yaml.safe_load(f)
        az_model_name = az_config['session_options']['model_name']
        az_model_version = str(az_config['session_options']['model_version'])

    if args.step == 'endpoint':
        endpoint = get_endpoint(ml_client, model)
        keys = None
        for i in range(10):
            try:
                keys = ml_client.online_endpoints.get_keys(endpoint_name)
                if keys.primary_key:
                    print('Successfully retrieved auth_key from endpoint')
                    break
            except ResourceNotFoundError:
                pass
            print(f'Waiting for endpoint to be created... ({i})')
            time.sleep(10)
        if not keys.primary_key:
            print('Cannot get auth_key from the endpoint. Please try running the script later.')
            exit(1)
        az_config['auth_key'] = keys.primary_key
        endpoint = get_endpoint(ml_client, model)
        scoring_uri = endpoint.scoring_uri
        if not scoring_uri:
            scoring_uri = f'https://{endpoint_name}.{endpoint.location}.inference.ml.azure.com'
        else:
            scoring_uri = scoring_uri.rstrip('/')
            if scoring_uri.endswith('/score'):
                scoring_uri = '/'.join(scoring_uri.split('/')[:-1])
        az_config['session_options']['uri'] = scoring_uri
        with open(p / 'azure_config.yaml', 'w') as f:
            yaml.dump(az_config, f)
        print(f'Endpoint information has been successfully written to {str(p / "azure_config.yaml")}')
    
    if args.step == 'deployment':
        model_path = prepare_model_path(p, az_model_name, az_model_version)
        
        endpoint = get_endpoint(ml_client, model)
        if not endpoint.provisioning_state == 'Succeeded':
            print(f'The endpoint {endpoint_name} is not ready yet.')
            print(f'Expected state: Succeeded. Current state: {endpoint.provisioning_state}')
            exit(1)
        deployment = ManagedOnlineDeployment(
            name = model,
            endpoint_name=endpoint_name,
            model = Model(
                name = az_model_name, 
                version = az_model_version,
                path = model_path.name,
                type="triton_model"
            ),
            instance_type = instance_type,
            instance_count = instance_count,
        )

        ml_client.online_deployments.begin_create_or_update(deployment)
        model_path.cleanup()

    if args.step == 'set_traffic':
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {model: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint)

    return


if __name__ == '__main__':
    args = parse_args()
    global EP_PREFIX
    EP_PREFIX = args.prefix
    ml_client = get_client(args.config)
    deploy(ml_client, args)
