import yaml

def load_config(config_file):
    """
    Load the YAML configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    # Convert memory strings to numerical values for comparison
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

        # Iterate through the flavors in reverse priority (higher to lower)
        for flavor in sorted(config['flavors'], key=lambda x: x['score_range'][0], reverse=True):
            score_range = flavor['score_range']
            required_cpu = float(flavor['resource_requirements']['cpu'])
            required_memory = parse_memory(flavor['resource_requirements']['memory'])

            # Check if score fits in the range
            if score_range[0] <= score <= score_range[1]:
                # Check if resources are sufficient
                if available_cpu >= required_cpu and available_memory_gi >= required_memory:
                    assigned_flavor = flavor['name']
                    break

        # Fallback to a lower flavor based on available resources, starting from the highest flavor
        if assigned_flavor == "Insufficient Resources":
            for flavor in sorted(config['flavors'], key=lambda x: (float(x['resource_requirements']['cpu']), parse_memory(x['resource_requirements']['memory'])), reverse=True):
                required_cpu = float(flavor['resource_requirements']['cpu'])
                required_memory = parse_memory(flavor['resource_requirements']['memory'])

                # Check if resources match for the current flavor
                if available_cpu >= required_cpu and available_memory_gi >= required_memory:
                    assigned_flavor = flavor['name']
                    break

        calibrated_flavors.append(assigned_flavor)

    return calibrated_flavors



if __name__ == "__main__":
    # Example configuration file
    config_file = "/home/ilias/Desktop/safe-6g/Cognitive/Code/repo/Cognitive_Coordinator/TF_Configuration_Files/reliability.yaml"

    # Load configuration
    config = load_config(config_file)

    # Example inputs
    scores = [35, 45, 85, 15, 95] 
    available_cpu = 1.5  # Available CPU resources
    available_memory = "1Gi"  # Available memory resources

    # Calibrate scores
    results = calibrate_scores(scores, config, available_cpu, available_memory)

    # Output results
    for score, flavor in zip(scores, results):
        print(f"Score: {score}, Assigned Flavor: {flavor}")
