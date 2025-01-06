import tensorflow as tf

def check_gpu():
    print("TensorFlow Version:", tf.__version__)

    # List all devices
    devices = tf.config.list_physical_devices()
    print("Physical devices:")
    for device in devices:
        print(device)

    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
    else:
        print("GPU is not available")

    # Test TensorFlow GPU operation
    try:
        with tf.device('/GPU:0'):  # Specify GPU device
            a = tf.constant([1.0, 2.0, 3.0, 4.0])
            b = tf.constant([2.0, 2.0, 2.0, 2.0])
            c = a + b
            print("TensorFlow can run on GPU")
            print("Result of GPU operation:", c.numpy())
    except RuntimeError as e:
        print("Error using GPU:", e)

check_gpu()