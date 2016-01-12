import os

import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from config.config_holder import ConfigHolder
from statistical_classifier.statistical_clf import prediction_scores, get_printable_scores_str


def create_network_model(input_size, out_size, model_path=None):
    # print input_size, out_size
    model = Sequential()
    h1 = 2048
    h2 = 1024
    # model.add(Dense(input_size, out_size, activation='softmax'))
    model.add(Dense(int(input_size), h1, activation='tanh'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(h1, h2, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(h2, out_size, activation='softmax'))
    if model_path is not None and os.path.isfile(model_path):
        model.load_weights(model_path)
    sgd = SGD(lr=0.1, momentum=0.9, decay=1.e-5, nesterov=True)
    # model.compile(loss='mse', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    # model.compile(loss=gps_loss, optimizer=sgd)
    print model.get_config(verbose=0)

    return model


def get_scores(nn, x_train, x_test, y_train, y_test):
    # clf.fit(x_train, y_train)
    # train_scores = prediction_scores(y_train, nn.predict(x_train))
    test_scores = prediction_scores(y_test, nn.predict(x_test))
    # print train_scores
    # cPickle.dump(clf, open(path_to_dump, "w+"))

    return {'accuracy': test_scores['accuracy'],
            'f1_score': test_scores['f1'],
            'precision': test_scores['precision'],
            'recall': test_scores['recall'],
            # 'accuracy_train': train_scores['accuracy'],
            # 'f1_score_train': train_scores['f1'],
            # 'precision_train': train_scores['precision'],
            # 'recall_train': train_scores['recall']
            }


if __name__ == "__main__":
    config = ConfigHolder()

    batch_size = 1024

    data = np.load(config["train_class_cleaned_part_npy"])
    x = data[:, :-1]
    y = data[:, -1]
    del data
    ss = StandardScaler()
    x = ss.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    lb = LabelBinarizer()
    lb.fit(y)

    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    true_labels = y_test
    # print train_y
    # y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]
    # train_x_dir = os.path.join(config["test_call_chunks"], "x")
    # train_y_dir = os.path.join(config["test_call_chunks"], "y")
    # model_path = config["nn_model"]
    # print x_train.shape
    # print y_train[0]
    model_path = config["nn_model"]
    # print len(np.unique(y))
    model = create_network_model(input_size=len(x[0]), out_size=len(np.unique(y)), model_path=model_path)

    # for train_x, train_y in train_gen(train_x_dir, train_y_dir):
    #     # print train_x.shape
    #     # print train_y.shape
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=10, validation_split=0.2)
    model.save_weights(model_path,overwrite=True)
    print "fitted"
    # score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=2)

    scores = get_scores(model, x_train, x_test, y_train, y_test)
    print get_printable_scores_str(scores)
