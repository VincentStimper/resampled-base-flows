import yaml

def get_config(path):
    """
    Read configuration parameter form file
    :param path: Path to the yaml configuration file
    :return: Dict with parameter
    """

    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)