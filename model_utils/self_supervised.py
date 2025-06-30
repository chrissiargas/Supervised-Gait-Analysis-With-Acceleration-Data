import os.path

from pre_processing.building import ssl_builder
from config.config_parser import Parser
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import shutil
from model_utils.model import contrastive_model
from model_utils.metrics import Metrics

def make_files(config):
    log_dir = os.path.join('logs', 'ssl_' + config.architecture + '_TB')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    try:
        shutil.rmtree(log_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    model_args = config.as_str(learning='ssl')
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

def ssl_train_evaluate(data: ssl_builder, summary: bool = False, verbose: bool = False):
    config = Parser()
    config.get_args()

    train, test, val = data()

    model = contrastive_model(data)
    model.compile()
    model.build_model(data.input_shape[1:])
    model.summary()

    log_dir, model_file = make_files(config)

    train_steps = data.train_batches
    test_steps = data.test_batches
    val_steps = data.val_batches

    tensorboard = TensorBoard(log_dir, histogram_freq=1)
    save_model = ModelCheckpoint(
        filepath = model_file,
        monitor = 'val_c_loss',
        verbose = 0,
        save_best_only = True,
        mode = 'min',
        save_weights_only = True
    )
    early_stopping = EarlyStopping(
        monitor = 'val_c_loss',
        min_delta = 0,
        patience = 20,
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

    model.evaluate(test, steps=test_steps, verbose=0)

if __name__ == '__main__':
    data = ssl_builder(generate=False)
    ssl_train_evaluate(data, summary=True, verbose=False)
    del data









