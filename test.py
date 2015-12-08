import pandas as pd

df = pd.read_csv(open('/home/marek/tmp/train.csv'))

taxies = set()
for taxi in df["TAXI_ID"]:
    taxies.add(taxi)

coordinateCount = 0
for cdList in df["POLYLINE"]:
    coordinateCount += len(cdList)

missingData = 0
for missing in df["MISSING_DATA"]:
    if (missing == True): missingData += 1

print "trips: ", len(df)
print "totalCoordinates: ", coordinateCount
print 'taxi count: ', len(taxies)
print 'trips with missing data: ', missingData
print 'avg trips per taxi: ', len(df) / len(taxies)
print 'avg coordinates per taxi: ', coordinateCount / len(taxies)
print 'avg coordinates per trip: ', coordinateCount / len(df)
