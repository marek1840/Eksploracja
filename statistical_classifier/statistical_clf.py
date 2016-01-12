import os

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config.config_holder import ConfigHolder


def get_scores(clf, x_train, x_test, y_train, y_test, path_to_dump, dump=True):
    if dump:
        clf.fit(x_train, y_train)

        print "Fit is done"
        # y_train_pred = clf.predict(x_train)
        # y_test_pred = clf.predict(x_test)
        # print "Prediction is done"
        # train_scores = prediction_scores(y_train, clf.predict(x_train))
        # cPickle.dump(clf, open(path_to_dump, "w+"))
        joblib.dump(clf, path_to_dump)
        print "pickle is done"
    else:
        clf = joblib.load(os.path.join(config["models_dir"], name + ".pkl"))
    test_scores = prediction_scores(y_test, clf.predict(x_test))

    return {
        'accuracy': test_scores['accuracy'],
        'f1_score': test_scores['f1'],
        'precision': test_scores['precision'],
        'recall': test_scores['recall']}
    # 'accuracy_train': train_scores['accuracy'],
    # 'f1_score_train': train_scores['f1'],
    # 'precision_train': train_scores['precision'],
    # 'recall_train': train_scores['recall']}


def prediction_scores(y_true, y_predicted):
    avg = 'weighted'
    accuracy = accuracy_score(y_true, y_predicted)
    f1 = f1_score(y_true, y_predicted, average=avg)
    precision = precision_score(y_true, y_predicted, average=avg)
    recall = recall_score(y_true, y_predicted, average=avg)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}


def get_labels_stats(labels):
    stats = {}
    for label in labels:
        if label in stats.keys():
            stats[label] += 1
        else:
            stats[label] = 1
    return stats


def get_printable_scores_str(scores):
    return "\taccuracy,\t test: %f\n" % (scores['accuracy']) + \
           "\tprecision,\t test: %f\n" % (scores['precision']) + \
           "\trecall,\t test: %f\n" % (scores['recall']) + \
           "\tf1 score,\t test: %f\n" % (scores['f1_score'])


if __name__ == "__main__":

    config = ConfigHolder()
    log_file = open(os.path.join(config["models_dir"], "log.txt"), "w+")
    neighbours_size = xrange(1, 10)

    classifiers = [
        # SVC(),
        RandomForestClassifier(max_depth=15, max_leaf_nodes=20, max_features=10),
        DecisionTreeClassifier(),
        # AdaBoostClassifier(),
        # GaussianNB()
    ]

    names = [
        # 'SVC',
        'RandomForest',
        'DecisionTree',
        # 'AdaBoost',
        # 'NaiveBayes'
    ]

    # classifiers.extend([KNeighborsClassifier(k) for k in neighbours_size])
    # names.extend(['KNN-%d' % k for k in neighbours_size])

    data = np.load(config["train_class_cleaned_part_npy"])
    x = data[:, :-1]
    y = data[:, -1]
    del data
    ss = StandardScaler()
    x = ss.fit_transform(x)

    best_clf = None
    best_avg_f1 = 0.
    best_scores = None

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    print 'Training data shape: ', x_train.shape, y_train.shape
    print 'Test data shape: ', x_test.shape, y_test.shape
    # del x_train
    # del y_train
    # x_train = 0
    # y_train = 0
    # dump = False
    dump = True
    for clf, name in zip(classifiers, names):
        print clf.__repr__()
        log_file.write(clf.__repr__())
        try:
            scores = get_scores(clf, x_train, x_test, y_train, y_test,
                                os.path.join(config["models_dir"], name + ".pkl")
                                , dump=dump)
            log_file.write(get_printable_scores_str(scores))
            # log_file.write('==================================================================')
            print scores
            if scores['f1_score'] > best_avg_f1:
                best_clf = clf
                best_scores = scores
        except MemoryError:
            log_file.write("Memory error")
            print "Memory error"
        print '=================================================================='
        log_file.write('\n==================================================================\n')

    print "Best classifier: %s" % best_clf.__repr__()
    # get_printable_scores_str(best_scores)

    log_file.write("Best classifier: %s" % best_clf.__repr__())
    log_file.write(get_printable_scores_str(best_scores))
    # best_clf.fit(x, y)
    # with open(config['classifier'], 'wb') as f:
    #     cPickle.dump(best_clf, f)
