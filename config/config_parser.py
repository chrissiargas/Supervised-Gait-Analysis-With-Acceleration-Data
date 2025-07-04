import argparse
import os.path
import yaml
from os.path import dirname, abspath
from typing import Optional

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
            help='pre_processing & training arguments'
        )

    def as_str(self, theme: Optional[str] = None, learning: str = 'sl', with_event: bool = True) -> str:

        if learning == 'sl':
            if theme is None:
                out = f'{self.test_hold_out}-'

            elif theme == 'preprocessing':
                out = (f'{self.test_hold_out}-'
                       f'{self.orient_method}-'
                       f'{self.include_g}-'
                       f'{self.features}-'
                       f'{self.filter}-'
                       f'{self.duration}-'
                       f'{self.stride}-')

            elif theme == 'task':
                out = (f'{self.test_hold_out}-'
                       f'{self.task}-'
                       f'{self.targets}-'
                       f'{self.target_oversampling}')

            elif theme == 'model':
                out = (f'{self.test_hold_out}-'
                       f'{self.architecture}-'
                       f'{self.head}-'
                       f'{self.loss}')

            else:
                out = (f'{self.test_hold_out}-'
                       f'{theme}')

            if with_event:
                out = (f'{str(self.labels)}-'
                       f'{out}')

            out = 'sl-' + out

        elif learning == 'ssl':
            if theme is None:
                out = (f'{self.task}-'
                       f'{self.targets}-'
                       f'{str(self.labels)}-'
                       f'{self.test_hold_out}-')

            out = 'ssl-' + out

        return out



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

        self.position = args.data_args['position']
        self.activities = args.data_args['activities']
        self.fs = args.data_args['fs']
        self.path = args.data_args['path']
        self.dataset = args.data_args['dataset']

        self.checks = args.main_args['checks']
        self.orient_method = args.main_args['orient_method']
        self.include_g = args.main_args['include_gravity']
        self.g_cutoff = args.main_args['gravity_cutoff']
        self.new_features = args.main_args['new_features']
        self.filter = args.main_args['filter']
        self.filter_cutoff = args.main_args['filter_cutoff']
        self.trim_duration = args.main_args['trim_duration']
        self.trim_length = int(self.trim_duration * self.fs)

        self.split_type = args.main_args['split_type']
        self.test_hold_out = args.main_args['test_hold_out']
        self.validation = args.main_args['validation']
        self.val_hold_out = args.main_args['val_hold_out']

        self.duration = args.main_args['duration']
        self.length = int(self.duration * self.fs)
        self.stride = args.main_args['stride']
        self.step = int(self.stride * self.fs) if self.stride != 0 else 1

        self.labels = args.main_args['labels']
        self.task = args.main_args['task']
        self.targets = args.main_args['targets']
        self.target_position = args.main_args['target_position']
        self.target_oversampling = args.main_args['target_oversampling']

        self.features = args.main_args['features']
        self.augmentations = args.main_args['augmentations']
        self.xyz = args.main_args['xyz']

        self.architecture = args.main_args['architecture']
        self.rotation_layer = args.main_args['rotation_layer']
        self.epochs = args.main_args['epochs']
        self.optimizer = args.main_args['optimizer']
        self.learning_rate = float(args.main_args['learning_rate'])
        self.head = args.main_args['head']
        self.class_weights = args.main_args['class_weights']
        self.loss = args.main_args['loss']
        self.metric = args.main_args['metric']

        self.anchor = args.main_args['anchor']
        self.target = args.main_args['target']
        self.batch_method = args.main_args['batch_method']
        self.batch_size = args.main_args['batch_size']
        self.neg_sub = args.main_args['negative_subject']
        self.neg_pos = args.main_args['negative_position']
        self.same_sub = True if self.neg_sub == 'same' else False


