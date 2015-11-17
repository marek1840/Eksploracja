import pandas as pd
import functools
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(file('/home/marek/tmp/train.csv'))

routes = [e for e in df["POLYLINE"]]

lengths = map(lambda seq: len(seq), routes)

plt.hist(lengths, bins=[x for x in xrange(0, 4000, 100)])
plt.show()