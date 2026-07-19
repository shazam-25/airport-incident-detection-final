import yaml

def get_metadata_from_yaml(yaml_path):
    """Reads a standard YOLO data.yaml configuration file
    and return the list of class names dynamically.
    """
    with open(yaml_path, 'r') as stream:
        try:
            data_config = yaml.safe_load(stream)
            # YOLO configs save class names under the key 'names'
            class_names = data_config.get('names', [])

            # Handle dictionary style class formats {0: 'class A', 1: 'class B'}
            if isinstance(class_names, dict):
                class_names = [class_names[i] for i in sorted(class_names.keys())]
            return class_names
        except yaml.YAMLError as exc:
            print(f"⚠️Error reading YAML file: {exc}")
            return []