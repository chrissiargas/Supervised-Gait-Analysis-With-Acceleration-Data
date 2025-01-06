import argparse
import os.path

import yaml
from os.path import dirname, abspath

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="pre-processing and training parameters"
        )

    def __call__(self, *args, **kwargs):
        project_root = dirname(abspath(__file__))
        config_path = os.path.join(project_root, 'config.yaml')

        self.parser.add_argument(
            '--config',
            default=config_path,
            help='config file location'
        )

        self.parser.add_argument(
            '--data_args',
            default=dict(),
            type=dict,
            help='data extraction arguments'
        )

        self.parser.add_argument(
            '--main_args',
            default=dict(),
            type=dict,
            help='preprocessing & training arguments'
        )


    def get_args(self):
        self.__call__()
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            defaultArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for defaultKey in defaultArgs.keys():
            if defaultKey not in keys:
                print('WRONG ARG: {}'.format(defaultKey))
                assert (defaultKey in keys)

        self.parser.set_defaults(**defaultArgs)
        args = self.parser.parse_args(args=[])

        self.load_data = args.main_args['load_data']

        self.position = args.data_args['position']
        self.activities = args.data_args['activities']
        self.fs = args.data_args['fs']
        self.path = args.data_args['path']

        self.cleaner = args.main_args['cleaner']
        self.include_g = args.main_args['include_gravity']
        self.new_features = args.main_args['new_features']
        self.filter = args.main_args['filter']
        self.filter_window = args.main_args['filter_window']
        self.rescaler = args.main_args['rescaler']
        self.parameters = args.main_args['parameters']
        self.task = args.main_args['task']

        self.split_type = args.main_args['split_type']
        self.test_hold_out = args.main_args['test_hold_out']
        self.validation = args.main_args['validation']
        self.val_hold_out = args.main_args['val_hold_out']

        self.duration = args.main_args['duration']
        self.length = int(self.duration * self.fs)
        self.stride = args.main_args['stride']
        self.step = int(self.stride * self.fs) if self.stride != 0 else 1

        self.features = args.main_args['features']
        self.augmentations = args.main_args['augmentations']

        self.batch_size = args.main_args['batch_size']

        self.architecture = args.main_args['architecture']
        self.epochs = args.main_args['epochs']
        self.optimizer = args.main_args['optimizer']
        self.learning_rate = args.main_args['learning_rate']

        self.metrics = args.main_args['metrics']

