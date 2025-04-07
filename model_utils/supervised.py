import os.path

from pre_processing.building import builder
from config.config_parser import Parser
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import shutil
from model_utils.model import alligaitor
from model_utils.metrics import Metrics

def make_files(config):
    log_dir = os.path.join('logs', config.architecture + '_TB')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    try:
        shutil.rmtree(log_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    model_args = f'{config.task}-{config.targets}-{str(config.labels)}'
    model_dir = os.path.join('archive', 'model_weights', model_args)
    model_file = '%s.weights.h5' % config.architecture
    model_file = os.path.join(model_dir, model_file)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    try:
        os.remove(model_file)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    return log_dir, model_file

def train_evaluate(data: builder, summary: bool = False, verbose: bool = False):
    config = Parser()
    config.get_args()

    train, test, val = data()

    model = alligaitor(data)
    model.compile()
    model.build_model(data.input_shape)
    model.summary()

    train_steps = data.train_size // config.batch_size
    test_steps = data.test_size // config.batch_size
    val_steps = data.val_size // config.batch_size

    log_dir, model_file = make_files(config)

    tensorboard = TensorBoard(log_dir, histogram_freq=1)
    save_model = ModelCheckpoint(
        filepath = model_file,
        monitor = 'val_loss',
        verbose = 0,
        save_best_only = True,
        mode = 'min',
        save_weights_only = True
    )
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 5,
        mode = 'min',
        verbose = 1
    )

    callbacks = [
        tensorboard,
        save_model,
        early_stopping
    ]

    model.fit(
        train,
        epochs = config.epochs,
        steps_per_epoch = train_steps,
        validation_data = val,
        validation_steps = val_steps,
        callbacks = callbacks,
        verbose = 1
    )

    model.load_weights(model_file)

    scores = ['accuracy','f1_score', 'precision', 'recall']
    test_metrics = Metrics(test, test_steps, log_dir, on='test_end', scores=scores, verbose=0)
    model.evaluate(test, steps=test_steps, callbacks=[test_metrics], verbose=0)









