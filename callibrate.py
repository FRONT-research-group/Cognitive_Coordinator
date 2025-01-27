import yaml
import requests
from pyswip import Prolog

def load_config(config_file):
    """
    Load the YAML configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_available_resources(api_url, token):
    """
    Fetch available resources (CPU and Memory) from the API.
    
    Args:
        api_url (str): The API endpoint.
        token (str): The bearer token for authentication.
    
    Returns:
        list of dict: A list of dictionaries containing available CPU and memory for each infrastructure element.
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        infrastructure_elements = response.json()
        resources = []
        for element in infrastructure_elements:
            available_cpu = element['cpuCores'] * (1 - element['currentCpuUsage'] / 100)
            available_memory = f"{element['availableRam'] / 1024:.2f}Gi"  # Convert MiB to GiB
            resources.append({
                "hostname": element['hostname'],
                "available_cpu": available_cpu,
                "available_memory": available_memory
            })
        return resources
    else:
        raise Exception(f"Failed to fetch resources: {response.status_code}, {response.text}")

def calibrate_scores(scores, config, available_cpu, available_memory):
    """
    Calibrate scores based on the configuration file and available resources.
    
    Args:
        scores (list): A list of trustworthiness scores to be calibrated.
        config (dict): The configuration loaded from the YAML file.
        available_cpu (float): Available CPU resources.
        available_memory (str): Available memory resources (e.g., "1Gi").

    Returns:
        list: A list of flavors assigned to each score.
    """
    def parse_memory(memory_str):
        if memory_str.endswith("Mi"):
            return float(memory_str[:-2]) / 1024  # Convert Mi to Gi
        elif memory_str.endswith("Gi"):
            return float(memory_str[:-2])
        else:
            raise ValueError(f"Unsupported memory format: {memory_str}")

    available_memory_gi = parse_memory(available_memory)
    calibrated_flavors = []

    for score in scores:
        assigned_flavor = "Insufficient Resources"
        for flavor in sorted(config['flavors'], key=lambda x: x['score_range'][0], reverse=True):
            score_range = flavor['score_range']
            required_cpu = float(flavor['resource_requirements']['cpu'])
            required_memory = parse_memory(flavor['resource_requirements']['memory'])
            if score_range[0] <= score <= score_range[1]:
                if available_cpu >= required_cpu and available_memory_gi >= required_memory:
                    assigned_flavor = flavor['name']
                    break
        if assigned_flavor == "Insufficient Resources":
            for flavor in sorted(config['flavors'], key=lambda x: (float(x['resource_requirements']['cpu']), parse_memory(x['resource_requirements']['memory'])), reverse=True):
                required_cpu = float(flavor['resource_requirements']['cpu'])
                required_memory = parse_memory(flavor['resource_requirements']['memory'])
                if available_cpu >= required_cpu and available_memory_gi >= required_memory:
                    assigned_flavor = flavor['name']
                    break
        calibrated_flavors.append(assigned_flavor)

    return calibrated_flavors

def load_prolog_knowledge(base_file):
    prolog = Prolog()
    prolog.consult(base_file)
    return prolog

def get_flavor(prolog, score, cpu, memory):
    query = list(prolog.query(f"assign_flavor({score}, {cpu}, {memory}, Flavor)"))
    if query:
        return query[0]["Flavor"]

    # Fallback logic
    fallback_query = list(prolog.query(f"fallback_flavor({cpu}, {memory}, Flavor)"))
    if fallback_query:
        return fallback_query[0]["Flavor"]

    return "Insufficient Resources"

if __name__ == "__main__":
    # Example configuration file
    config_file = "/home/ilias/Desktop/safe-6g/Cognitive/Code/repo/Cognitive_Coordinator/TF_Configuration_Files/reliability.yaml"
    api_url = "https://safe-6g-ncsrd.satrd.es/entities?type=InfrastructureElement&format=simplified"
    token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJzaTcxSzNkUm11UFIxY2RhT2daNVFtbGpUVlR6U3JQM0cyYlZNdEVDeUVjIn0.eyJleHAiOjE4MDk0MjgwMDMsImlhdCI6MTcyMzExNDQwMywianRpIjoiMzI4OTdiMWEtNjFjMy00Yzk0LTkwN2QtZDg0Y2Y1NTQwOTNhIiwiaXNzIjoiaHR0cHM6Ly9rZXljbG9hay5jZi1tdnAtZG9tYWluLmFlcm9zLXByb2plY3QuZXUvYXV0aC9yZWFsbXMva2V5Y2xvYWNrLW9wZW5sZGFwIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6IjU5MzMyMDg2LTA0YzgtNGFiZS1iY2JiLWEyZjMzMmM3M2FmYSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImFlcm9zLXRlc3QiLCJzZXNzaW9uX3N0YXRlIjoiN2YyOWMyMzEtMWY2Zi00YzNkLThhMDEtMjVmZjVmMjMwNDA2IiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJDbG91ZEZlcnJvRG9tYWluIiwiZGVmYXVsdC1yb2xlcy1rZXljbG9hY2stb3BlbmxkYXAiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiRG9tYWluIGFkbWluaXN0cmF0b3IiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiI3ZjI5YzIzMS0xZjZmLTRjM2QtOGEwMS0yNWZmNWYyMzA0MDYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsIm5hbWUiOiJEb21haW4gYWRtaW5pc3RyYXRvciAxIEFkbWluIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiZG9tYWluYWRtaW5pc3RyYXRvcjEiLCJnaXZlbl9uYW1lIjoiRG9tYWluIGFkbWluaXN0cmF0b3IgMSIsImZhbWlseV9uYW1lIjoiQWRtaW4ifQ.dF3CS5Wq23YB2sbc2-epH4QLhN9Y9JGuqW7aQ5x0mGgM4v_bkmycXnceuKohVxgSreSq5jJ-m7P38-HGfX0GoLDiSENqlw8SzMyKmjlFuu5rreXIbskI3GKqbfGog4ZR8ojTCCfbfwgdZsvc_XFZRnrsC_nFuHe2AiD3ypWPFnEY9edvzG-oWC414hvIHGLdVAXqLthWJe65s1QfOWrn70lvBuszHDg48iec_zv0Us5u8yeYXahO8Tf7FrQ4CgGuocS2vn55ENQgLDs03E01m6CWPlANhgJKEfziPGCxRuYKIDNZOrvhIF-ZsMEsrt95jg-qeskqkdA2dWzbJwM5Sw"

    # Load configuration
    #config = load_config(config_file)
    prolog = load_prolog_knowledge("TF_Configuration_Files/reliability.pl")

    # Fetch available resources
    resources = get_available_resources(api_url, token)

    # Print all available resources
    for resource in resources:
        print(f"Hostname: {resource['hostname']}, CPU: {resource['available_cpu']:.2f}, Memory: {resource['available_memory']}")

    # Sort resources by available CPU and memory in descending order
    resources_sorted = sorted(
        resources,
        key=lambda r: (r['available_cpu'], float(r['available_memory'][:-2])),  # Convert memory to float for sorting
        reverse=True
    )

    # Select the infrastructure element with the most free resources
    best_resource = resources_sorted[0]
    available_cpu = best_resource['available_cpu']
    available_memory = best_resource['available_memory']
    #available_memory = '600'
    print(f"\nUsing best resource: Hostname: {best_resource['hostname']}, CPU: {available_cpu:.2f}, Memory: {available_memory}")

    # Example inputs
    scores = [35, 45, 85, 15, 95]

    # Calibrate scores
    #results = calibrate_scores(scores, config, available_cpu, available_memory)
    
    # Output results
    # for score, flavor in zip(scores, results):
    #     print(f"Score: {score}, Assigned Flavor: {flavor}")

    # Convert available_memory to MB once
    if available_memory.endswith('Gi'):
        available_memory = float(available_memory.replace('Gi', '')) * 1024
    elif available_memory.endswith('Mi'):
        available_memory = float(available_memory.replace('Mi', ''))

    # Process scores
    for score in scores:
        flavor = get_flavor(prolog, score, available_cpu, available_memory)
        print(f"Score: {score}, Assigned Flavor: {flavor}")
