from tensorboard import program
model = 'sl_cnn-gru'

tracking_address = 'logs' + '/' + model + '_TB'  # the path of your log file. (accEncoderTb, fullModelTb)
if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()
