import cPickle
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot

from config.config_holder import ConfigHolder

conf = ConfigHolder()


class Vizualize():
    def plot_hisotgram(self, paths=False):

        if paths:
            # reading training data
            # zf = zipfile.ZipFile('/home/marek/tmp/train.csv.zip')
            df = pd.read_csv(conf["train"], converters={'POLYLINE': lambda x: json.loads(x)[-1:]})
            latlong = np.array([[p[0][0], p[0][1]] for p in df['POLYLINE'] if len(p) > 0])
        else:
            latlong = np.array(cPickle.load(open(conf["mean_shift_cluster_centers"])))

        print 'read'
        # cut off long distance trips
        lat_low, lat_hgh = np.percentile(latlong[:, 0], [1, 90])
        lon_low, lon_hgh = np.percentile(latlong[:, 1], [1, 90])
        print 'cut'
        # create image
        bins = 256
        lat_bins = np.linspace(lat_low, lat_hgh, bins)
        lon_bins = np.linspace(lon_low, lon_hgh, bins)
        H2, _, _ = np.histogram2d(latlong[:, 1], latlong[:, 0], bins=(lon_bins, lat_bins))
        print 'created'
        img = np.log(H2[::-1, :] + 1)

        plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.axis('off')

        if paths:
            plt.title('Taxi trip end points')
            plt.savefig("taxi_trip_end_points.png")
        else:
            plt.title('Taxi trip clustered end points')
            plt.savefig("taxi_trip_clustered_end_points.png")
        print 'plotted'

    def plot_cluster_size(self):
        latlong = pd.read_csv(conf["train_class"])["CLASS"]
        bins = np.linspace(0, 3500, 3500)
        pyplot.hist(latlong, bins)
        # pyplot.show()
        pyplot.xlabel("Cluster ID")
        pyplot.ylabel("Number of localizations in cluster")
        pyplot.savefig("custer_number.png")




if __name__=='__main__':
    v = Vizualize()
    v.plot_cluster_size()