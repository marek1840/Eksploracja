import cPickle
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift

from config.config_holder import ConfigHolder


class MeanShiftClf:
    def get_clusters(self, in_file, cc_file, clf_file, arrivals_file, chunk_size=1710671):
        df = pd.read_csv(open(in_file), chunksize=chunk_size)
        dests = []
        part = 1
        lines = 1710671 / chunk_size
        try:
            dest = cPickle.load(open(arrivals_file))
        except IOError:
            for d in df:
                print "%d / %d" % (part, lines)
                part += 1
                for row in d.values:
                    # print eval(row[-1])
                    tmp = eval(row[-1])
                    if len(tmp) > 0:
                        dests.append(tmp[-1])
            dest = np.array(dests)
            cPickle.dump(dest, open(arrivals_file, "w"), protocol=cPickle.HIGHEST_PROTOCOL)
        print "Destination points loaded"

        try:
            ms = cPickle.load(open(clf_file))
        except IOError:
            bw = 0.001
            ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5, n_jobs=-2)
            ms.fit(dest)
            cPickle.dump(ms, open(clf_file, "w"), protocol=cPickle.HIGHEST_PROTOCOL)
        print "Mean shift loaded"
        cluster_centers = ms.cluster_centers_
        cPickle.dump(cluster_centers, open(cc_file, "w"), protocol=cPickle.HIGHEST_PROTOCOL)
        print "Clusters dumped"

    def add_class(self, in_file, out_file, clf_file):
        ms = cPickle.load(open(clf_file))
        labels = ms.labels_
        del ms

        df = pd.read_csv(open(in_file))
        df['CLASS'] = pd.Series(labels)
        # cut destination point (last in sequence) because it will give class explicite
        df['POLYLINE'] = df['POLYLINE'].apply(lambda x: x[:x.rfind('[') - 1] + "]" if len(x) > 0 else x)
        df.to_csv(open(out_file, "w+"), index=False)

    def clean(self, conf):
        to_delete = [conf["mean_shift_cluster_centers"], conf["mean_shift_model"], conf["arrivals"]]
        for i in to_delete:
            os.remove(i)


if __name__ == "__main__":
    conf = ConfigHolder()
    ms = MeanShiftClf()
    ms.get_clusters(conf["train"], cc_file=conf["mean_shift_cluster_centers"],
                    clf_file=conf["mean_shift_model"], arrivals_file=conf["arrivals"])
    ms.add_class(conf["train"], conf["train_class"], clf_file=conf["mean_shift_model"])
