import math
import os

import pandas as pd
from sklearn.cross_validation import train_test_split

from config.config_holder import ConfigHolder

__author__ = 'michal'


def split_to_balanced_train_test(in_csv_file, out_dir):
    f = pd.read_csv(in_csv_file)
    print f.columns.values
    # print f['CALL_TYPE'].val
    # print f.mask('CALL_TYPE', 'A')'A'
    # os.makedirs(out_dir)
    posible_val = ['A', 'B', 'C']
    out_train_call = pd.DataFrame()
    out_test_call = pd.DataFrame()

    out_train_day = pd.DataFrame()
    out_test_day = pd.DataFrame()
    for val in posible_val:
        train, test = train_test_split(f[(f.CALL_TYPE == val)], test_size=0.3)

        out_train_call = out_train_call.append(train)
        out_test_call = out_test_call.append(test)

        train, test = train_test_split(f[(f.DAY_TYPE == val)], test_size=0.3)

        out_train_day = out_train_day.append(train)
        out_test_day = out_test_day.append(test)
        # print f[(f.CALL_TYPE == val)]
        # print f[(f.DAY_TYPE == val)]
        # print f.filter(like="f['CALL_TYPE'] == 'B'")

    # print out_test
    # TODO save in file
    if not os.path.exists(os.path.join(out_dir, 'CALL_TYPE')):
        os.makedirs(os.path.join(out_dir, 'CALL_TYPE'))
    out_test_call.to_csv(os.path.join(out_dir, 'CALL_TYPE', 'test.csv'))
    out_train_call.to_csv(os.path.join(out_dir, 'CALL_TYPE', 'train.csv'))

    if not os.path.exists(os.path.join(out_dir, 'DAY_TYPE')):
        os.makedirs(os.path.join(out_dir, 'DAY_TYPE'))
    out_test_day.to_csv(os.path.join(out_dir, 'DAY_TYPE', 'test.csv'))
    out_train_day.to_csv(os.path.join(out_dir, 'DAY_TYPE', 'train.csv'))


def distance_between_gps(lat0, lon0, lat1, lon1):
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


if __name__ == "__main__":
    conf =ConfigHolder()
    in_path = conf["train"]

    split_to_balanced_train_test(in_path, "../data/splited")
