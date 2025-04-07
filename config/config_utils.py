import ruamel.yaml
from os.path import dirname, abspath, join

def config_edit(args, parameter, value):
    yaml = ruamel.yaml.YAML()

    project_root = dirname(abspath(__file__))
    config_path = join(project_root, 'config.yaml')

    with open(config_path) as fp:
        data = yaml.load(fp)

    for param in data[args]:

        if param == parameter:
            data[args][param] = value
            break

    with open(config_path, 'w') as fb:
        yaml.dump(data, fb)


def config_save(paramsFile):
    yaml = ruamel.yaml.YAML()

    with open('config_utils/config.yaml') as fp:
        parameters = yaml.load(fp)

    with open(paramsFile, 'w') as fb:
        yaml.dump(parameters, fb)