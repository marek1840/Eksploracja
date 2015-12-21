import os
from itertools import izip, count
from os import listdir

import numpy as np
import pandas as pd
import theano

from config.config_holder import ConfigHolder
from utils.io_utils import check_if_dir_exists_else_create


class CsvConverter(object):
    def __init__(self):
        self.config = ConfigHolder()

    def convert_file(self, infile, outfile):
        # df = pd.read_csv(open(infile))
        # print df.P
        x, y = self.get_raw_features_tensors(infile, int(self.config["window_size"]), int(self.config["chunk_size"]))


    def get_raw_features_tensors(self, path, dir_out, window_size=100, chunk_size=20000, input_size=50, label_size=2):

        df = pd.read_csv(open(path), chunksize=chunk_size)
        check_if_dir_exists_else_create(os.path.join(dir_out, "x"))
        check_if_dir_exists_else_create(os.path.join(dir_out, "y"))
        features = []
        labels_lstm = []
        current_chunck = 0
        part_number = 0
        for d in df:

            for row in d.values:

                # print row
                # for r in row:
                #     print r
                # print row
                # print row.CALL_TYPE
                CALL_TYPE = ord(row[3]) - ord('A')  # "CALL_TYPE"
                DAY_TYPE = ord(row[8]) - ord('A')  # "DAY_TYPE"
                # print
                # print(row[9])
                l = eval(row[10])
                for i in xrange(len(l) - window_size):
                    vec = [CALL_TYPE, DAY_TYPE]
                    vec.extend([j for sub in l[i:i + window_size]
                                for j in sub])

                    labels_lstm.append(l[i + window_size])
                    features.append(np.array(vec))
                    current_chunck += 1
                    if current_chunck == chunk_size:
                        print(part_number)
                        # print(features.shape)
                        # print(labels_lstm.shape)
                        last_dim = window_size * 2 + 2
                        x_tmp = np.asarray(features, dtype=theano.config.floatX)
                        # print x_tmp.shape
                        # print chunk_size, input_size, last_dim
                        x_tmp = x_tmp.reshape(chunk_size / input_size,
                                              input_size,
                                              last_dim)
                        # print x_tmp.shape
                        x_tmp.dump(
                            os.path.join(dir_out, "x", "part_%s.pkl" % part_number))
                        # print "x done"
                        tmp = np.asarray(labels_lstm, dtype=theano.config.floatX)
                        # print(tmp.shape)
                        tmp = tmp.reshape(chunk_size, label_size)
                        print tmp.shape
                        tmp.dump(
                            os.path.join(dir_out, "y", "part_%s.pkl" % part_number))
                        current_chunck = 0
                        features = []
                        labels_lstm = []
                        part_number += 1


    def get_raw_test(self, path, out, window_size=100):
        df = pd.read_csv(open(path))
        features = []
        for d, part_number in izip(df, count(0, 1)):
            print(part_number)
            for row in d.values:
                CALL_TYPE = ord(row[3]) - ord('A')  # "CALL_TYPE"
                DAY_TYPE = ord(row[8]) - ord('A')  # "DAY_TYPE"
                l = eval(row[10])[window_size:]  # list of positions
                vec = [CALL_TYPE, DAY_TYPE]
                vec.extend([j for j in l])
                features.append(vec)


    def conver_dir(self, indir, outdir, suffix=True):
        for path in listdir(indir):
            self.convert_file(os.path.join(indir, path),
                              os.path.join(outdir, path[:-4] + "converted.csv" if suffix else path))


if __name__ == "__main__":
    c = CsvConverter()
    # c.convert_file(c.config['head'], c.config['head'][:-4] + "converted.csv")
    # c.get_raw_features_tensors(c.config['head'], c.config['head'][:-4] + "converted.csv")
