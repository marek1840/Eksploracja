import math
import os
from itertools import count

import numpy as np
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD

from config.config_holder import ConfigHolder
from munging.csv_converter import CsvConverter

conf = ConfigHolder()


# TODO http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf
# https://www.me.utexas.edu/~jensen/ORMM/supplements/units/time_series/time_series.pdf
def distance_between_gps(lat0, lon0, lat1, lon1):
    # TODO rewrite as theano funciton
    # i.e. https://github.com/fchollet/keras/blob/master/keras/objectives.py
    r = 6378137.
    lat0_r = (lat0 * math.pi) / 180.
    lat1_r = (lat1 * math.pi) / 180.
    dlat = (lat0 - lat1) * math.pi / 180
    dlon = (lon0 - lon1) * math.pi / 180
    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat0_r) * math.cos(lat1_r) * \
        math.sin(dlon / 2.) ** 2
    c = 2. * math.atan2(a ** 0.5, (1 - a) ** 0.5)
    ret = r * c
    return ret


def gps_loss(y_true, y_pred):
    return distance_between_gps(y_true[0], y_true[1], y_pred[0], y_pred[1])


def gps_evaluation(model, X, y):
    predicted = model.predict(X)
    print predicted
    measure = map(lambda p, y: gps_loss(y, p), predicted, y)

    print "Sum"
    print sum(measure)
    print "Avrage"
    print sum(measure) / len(measure)


def create_network_model(input_size, out_size, model_path=None):
    model = Sequential()
    h1 = 1024
    h2 = 512
    h3 = 512
    model.add(LSTM(int(input_size), h1, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(h1, h2))
    # changes according to used dataset
    # model.add(Activation('tanh'))
    model.add(LSTM(h2, 128))
    model.add(Dropout(0.2))
    model.add(Dense(128, out_size))
    model.add(Activation('tanh'))
    if os.path.isfile(model_path):
        model.load_weights(model_path)
    sgd = SGD(lr=0.1, momentum=0.9, decay=1.e-5, nesterov=True)
    # model.compile(loss='mse', optimizer='rmsprop')
    model.compile(loss='mse', optimizer=sgd)
    # model.compile(loss=gps_loss, optimizer=sgd)
    model.get_config(verbose=0)

    return model


def train_gen(x_dir, y_dir):
    for x, y, n in zip(sorted(os.listdir(x_dir)), sorted(os.listdir(y_dir)), count(0, 1)):
        print x, y
        yield np.load(os.path.join(x_dir, x)), np.load(os.path.join(y_dir, y))


def test_gen(x_dir, y_dir):
    for x, y, n in zip(sorted(os.listdir(x_dir)), sorted(os.listdir(y_dir)), count(0, 1)):
        if n > 5:
            yield np.load(os.path.join(x_dir, x)), np.load(os.path.join(y_dir, y))


if __name__ == "__main__":
    conf = ConfigHolder()
    # TODO mean shift

    cc = CsvConverter()

    batch_size = 128
    input_size = 200

    # cc.get_raw_features_tensors(conf["test_call"], conf["test_call_chunks"],
    #                             window_size=conf["window_size"],
    #                             chunk_size=conf["chunk_size"],
    #                             input_size=input_size,
    #                             label_size=2)
    print "Converted"

    train_x_dir = os.path.join(conf["test_call_chunks"], "x")
    train_y_dir = os.path.join(conf["test_call_chunks"], "y")
    model_path = conf["lstm_model"]

    model = create_network_model(input_size=6, out_size=2, model_path=model_path)
    # print "Fitting model"#https://github.com/fchollet/keras/issues/85
    # for train_x, train_y in train_gen(train_x_dir, train_y_dir):
    #     # print train_x.shape
    #     # print train_y.shape
    #     model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=2)
    #     break
    model.save_weights(model_path, overwrite=True)
    for train_x, train_y in train_gen(train_x_dir, train_y_dir):
        # score = model.evaluate(train_x, train_y, show_accuracy=True, verbose=2)
        # print 'Train score: '
        # print score
        print 'Train gps score'
        print gps_evaluation(model, train_x, train_y)

    for test_x, test_y in test_gen(train_x_dir, train_y_dir):
        score = model.evaluate(test_x, test_y, show_accuracy=True, verbose=2)
        print 'Test score: '
        print score


        #

        # score = model.evaluate(test_x, test_y, show_accuracy=True, verbose=2)
        # print 'Test score: '
        # print score
        # score = model.evaluate(train_x, train_y, show_accuracy=True)
        # print 'Train score: '
        # print score
        # #
        # predictions = model.predict(test_x)
        #
        # precision, recall, accuracy, f_score = get_stats(test_y, predictions, len(lb.classes_))
        #
        # print 'Precision: '
        # print precision
        # print precision.mean()
        #
        # print 'Recall: '
        # print recall
        # print recall.mean()
        #
        # print 'Accuracy: '
        # print accuracy
        # print accuracy.mean()
        #
        # print 'f_score: '
        # print f_score
        # print f_score.mean()
