from keras import backend as K


def ctc_lambda_func(args):
    y_predicition, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_predicition = y_predicition[:, 2:, :]
    return K.ctc_batch_cost(labels, y_predicition, input_length, label_length)
