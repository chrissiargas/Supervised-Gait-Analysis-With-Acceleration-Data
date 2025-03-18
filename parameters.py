sl_params = {
    'checks': None,
    'orient_method': None,
    'include_gravity': True,
    'gravity_cutoff': 0.1,
    'new_features': None,
    'filter': None,
    'filter_cutoff': 10.0,
    'labels': ['LF_HS'],
    'task': 'gait_events',
    'targets': 'one',
    'target_position': 'center',
    'trim_duration': 5,
    'duration': 4,
    'stride': 1,
    'features': ['acc_x', 'acc_y', 'acc_z'],
    'split_type': 'loso',
    'test_hold_out': [107],
    'validation': False,
    'val_hold_out': [3],

    'batch_size': 128,
    'architecture': 'cnn-gru',
    'rotation_layer': None,
    'optimizer': 'adam',
    'epochs': 250,
    'decay_steps': 0,
    'lr_decay': None,
    'learning_rate': 0.0001,
    'head': 'single',
    'class_weights': True,
}

