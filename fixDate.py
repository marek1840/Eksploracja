import datetime

import pandas as pd

from config.config_holder import ConfigHolder

day = 3.6e6

dayTypes = {
    (2012, 12, 31): "C", (2013, 1, 1): "B",
    (2013, 2, 10): "C", (2013, 2, 11): "B",
    (2013, 2, 13): "C", (2013, 2, 14): "B",
    (2013, 3, 28): "C", (2013, 3, 29): "B",
    (2013, 3, 30): "C", (2013, 3, 31): "B",
    (2013, 4, 1): "B",
    (2013, 4, 24): "C", (2013, 4, 25): "B",
    (2013, 4, 30): "C", (2013, 5, 1): "B",
    (2013, 5, 4): "C", (2013, 5, 5): "B",
    (2013, 5, 8): "C", (2013, 5, 9): "B",
    (2013, 5, 29): "C", (2013, 5, 30): "B",
    (2013, 6, 9): "C", (2013, 6, 10): "B",
    (2013, 6, 12): "C", (2013, 6, 13): "B",
    (2013, 6, 23): "C", (2013, 6, 24): "B",
    (2013, 6, 28): "C", (2013, 6, 29): "B",
    (2013, 8, 14): "C", (2013, 8, 15): "B",
    (2013, 10, 4): "C", (2013, 10, 5): "B",
    (2013, 10, 31): "C", (2013, 11, 1): "B",
    (2013, 11, 03): "C", (2013, 12, 1): "B",
    (2013, 12, 7): "C", (2013, 12, 8): "B",
    (2013, 12, 24): "C", (2013, 12, 25): "B",

    (2013, 12, 31): "C", (2014, 1, 1): "B",
    (2014, 2, 13): "C", (2014, 2, 14): "B",
    (2014, 3, 2): "C", (2014, 3, 3): "B",
    (2014, 4, 17): "C", (2014, 4, 18): "B",
    (2014, 4, 19): "C", (2014, 4, 20): "B",
    (2014, 4, 21): "B",
    (2014, 4, 24): "C", (2014, 4, 25): "B",
    (2014, 4, 30): "C", (2014, 5, 1): "B",
    (2014, 5, 3): "C", (2014, 5, 4): "B",
    (2014, 5, 28): "C", (2014, 5, 29): "B",
    (2014, 6, 9): "C", (2014, 6, 10): "B",
    (2014, 6, 12): "C", (2014, 6, 13): "B",
    (2014, 6, 18): "C", (2014, 6, 19): "B",
    (2014, 6, 23): "C", (2014, 6, 24): "B",
    (2014, 6, 28): "C", (2014, 6, 29): "B",
    (2014, 8, 14): "C", (2014, 8, 15): "B",
    (2014, 10, 4): "C", (2014, 10, 5): "B",
    (2014, 10, 31): "C", (2014, 11, 1): "B",
    (2014, 11, 03): "C", (2014, 12, 1): "B",
    (2014, 12, 7): "C", (2014, 12, 8): "B",
    (2014, 12, 24): "C", (2014, 12, 25): "B"
}

print 'opening file'
conf = ConfigHolder()

files = [
    # conf['test_call'],
    # conf['train_call'],
    # conf['test_day'],
    # conf['train_day'],
    # conf['train'],
    # "/home/michal/Studia/9Semestr/EksploracjaDanych/Eksploracja/data/Porto_taxi_data_test_partial_trajectories.csv", #NAN
    "/home/michal/Studia/9Semestr/EksploracjaDanych/Eksploracja/data/Porto_taxi_data_training.csv",  # EOF
]

for f in files:

    df = pd.read_csv(open(f))

    print f
    total = len(df)
    for index, row in df.iterrows():
        timestamp = row[5]
        print(timestamp)
        date = datetime.datetime.fromtimestamp(int(timestamp))
        key = (date.year, date.month, date.day)
        if key in dayTypes:
            dayType = dayTypes[key]
            row[6] = dayType

    df.to_csv(f)
