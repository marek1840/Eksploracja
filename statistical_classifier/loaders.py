import re

import numpy as np
import pandas as pd
import theano

from config.config_holder import ConfigHolder


def find_last_n_points(text, n):
    """
        list in form of string
        and flatten list
        trick: counting form back get nth '['
    """
    # positions = re.findall(text)

    rev = text[::-1]
    last = 0
    for match, n in zip(re.finditer('\[', rev), xrange(n)):
        last = match.start()

    last += 1
    rev = rev[:last]
    # print ("[" if (rev.count("]") % 2 == 1) else "") + rev[::-1]
    rev = re.sub("\[|]", "", rev)
    return rev[::-1]


def clean_file(in_file, out_file):
    df = pd.read_csv(open(in_file))
    # print list(df.columns.values)
    # print len(df.POLYLINE)
    df = df[df['POLYLINE'].apply(lambda x: len(x) > 135)]
    df = df[df.MISSING_DATA != True]
    print len(df.POLYLINE)
    df.to_csv(open(out_file, "w+"), index=False)


def load_csv_to_numpy_arrays(path, out_path, delimiter=','):
    call_type = {'A': 1, 'B': 2, 'C': 3}
    n_size = 6

    df = pd.read_csv(open(path), delimiter=delimiter)
    df.drop('TRIP_ID', axis=1, inplace=True)
    # df.drop(0, axis=1, inplace=True)
    df.drop('ORIGIN_CALL', axis=1, inplace=True)
    df.drop('ORIGIN_STAND', axis=1, inplace=True)
    df.drop('TAXI_ID', axis=1, inplace=True)
    df.drop('TIMESTAMP', axis=1, inplace=True)
    df.drop('MISSING_DATA', axis=1, inplace=True)

    df['CALL_TYPE'] = df['CALL_TYPE'].apply(lambda x: call_type[x])
    df['DAY_TYPE'] = df['DAY_TYPE'].apply(lambda x: call_type[x])
    df['POLYLINE'] = df['POLYLINE'].apply(lambda x: find_last_n_points(x, n_size))

    df.to_csv(open(path + ".tmp", "w+"), index=False, delimiter=delimiter)

    # df = pd.read_csv(open(path + ".tmp"), delimiter=delimiter)

    positions = df['POLYLINE'].apply(lambda x: pd.Series(x.split(',')))
    classes = df['CLASS']
    newdf = df.drop('CLASS', axis=1).drop('POLYLINE', axis=1).join(positions).join(classes)

    np.save(out_path, np.asarray(newdf, dtype=theano.config.floatX))


if __name__ == "__main__":
    conf = ConfigHolder()
    # text = "[[0],[1],[2],[3],[4],[5],[6]]"
    # print find_last_n_points(text, 7)
    # clean_file(conf["train"], conf["train"])
    clean_file(conf["train_class"], conf["train_class_cleaned"])
    load_csv_to_numpy_arrays(conf["train_class_cleaned"], conf["train_class_cleaned_npy"])
    # print np.load(conf["train_class_cleaned"] + ".pkl.npy")
