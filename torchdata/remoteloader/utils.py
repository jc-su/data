from logging import warn
import socket
import requests
import json
import time
from torchdata.dataloader2.graph import find_dps
from kubernetes import client, config


def get_local_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def get_container_info(container_name):
    hostname = "localhost"
    port = 8080
    version = "v1.3"
    base_url = f"http://{hostname}:{port}/api/{version}"

    endpoint = f"/docker/{container_name}"
    response = requests.get(base_url + endpoint)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None


def get_container_metrics(container_name, start_time, end_time):
    prometheus_server = "localhost"
    queries = {
        "cpu_usage": f'container_cpu_usage_seconds_total{{container_label_com_docker_compose_service="{container_name}"}}',
        "memory_usage": f'container_memory_usage_bytes{{container_label_com_docker_compose_service="{container_name}"}}',
        "total_memory": f'container_memory_max_usage_bytes{{container_label_com_docker_compose_service="{container_name}"}}'
    }

    metrics_data = {}

    for query_name, query in queries.items():
        response = requests.get(f'http://{prometheus_server}:9090/api/v1/query_range',
                                params={'query': query, 'start': start_time, 'end': end_time, 'step': '1s'})
        data = response.json()
        metrics_data[query_name] = [float(item[1]) for item in data['data']['result'][0]['values']]

    return metrics_data


def get_machine_metrics(prometheus_server, start_time, end_time):
    # Metric queries
    queries = {
        "cpu_usage": f'(100 - (avg(irate(node_cpu_seconds_total{{mode="idle"}}[5m])) by (instance) * 100))',
        "memory_usage": f'(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100',
        "disk_io": f'rate(node_disk_io_time_seconds_total[5m])',
        "network_io": f'rate(node_network_transmit_bytes_total[5m])',
    }

    metrics_data = {}

    for query_name, query in queries.items():
        response = requests.get(f'http://{prometheus_server}:9090/api/v1/query_range',
                                params={'query': query, 'start': start_time, 'end': end_time, 'step': '1s'})
        data = response.json()
        metrics_data[query_name] = data

    return metrics_data


def find_dp(graph, dp_type):
    pipes = find_dps(graph, dp_type)
    if len(pipes) == 1:
        return pipes[0]
    elif len(pipes) > 1:
        found_ids = set([id(pipe) for pipe in pipes])
        if len(found_ids) > 1:
            warn(f"""There are {len(pipes)} pipes of type {dp_type}. If this is intended, 
                     please use `find_dps` directly. Returning first instance.""")
        return pipes[0]
    else:
        raise LookupError(f'Unable to find {dp_type} starting at {graph}')


def spawn_worker(worker_name, trainer_ip, trainer_port, controller_ip, controller_port):
    config.load_kube_config()
    api = client.CoreV1Api()
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=worker_name),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name=worker_name,
                    image="torchdata/worker:latest",
                    env=[
                        client.V1EnvVar(name="TRAINER_IP", value=trainer_ip),
                        client.V1EnvVar(name="TRAINER_PORT", value=trainer_port),
                        client.V1EnvVar(name="CONTROLLER_IP", value=controller_ip),
                        client.V1EnvVar(name="CONTROLLER_PORT", value=controller_port),
                    ],
                    ports=[client.V1ContainerPort(container_port=80)],
                )
            ]
        ),
    )
    api.create_namespaced_pod(namespace="default", body=pod)


def delete_worker(worker_name):
    api = client.CoreV1Api()
    api.delete_namespaced_pod(namespace="default", name=worker_name)


def get_service_ip_and_ports(service_name, namespace):
    api = client.CoreV1Api()
    service = api.read_namespaced_service(name=service_name, namespace=namespace)
    ip = service.spec.cluster_ip
    ports = [port.port for port in service.spec.ports]
    return ip, ports


if __name__ == "__main__":
    end_time = time.time()
    start_time = end_time - 20
    print(get_container_metrics("redis", start_time, end_time))
    # print(get_machine_metrics("localhost", start_time, end_time))
