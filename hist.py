import matplotlib.pyplot as plt
import pandas as pd

from config.config_holder import ConfigHolder

conf = ConfigHolder()
df = pd.read_csv(open(conf['train']))

routes = [e for e in df["POLYLINE"]]

lengths = map(lambda seq: len(seq), routes)

plt.hist(lengths, bins=[x for x in xrange(0, 4000, 100)])
plt.savefig('hist.png')