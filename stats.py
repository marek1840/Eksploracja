import pandas as pd
df = pd.read_csv(file('/home/marek/tmp/train.csv'))

taxies = set()

total = 0
maxLen = 0
minLen = 10000
for route in df["POLYLINE"]:
	l = len(route)
	total += l
	if l > maxLen: 
		maxLen = l
	if l < minLen:
		minLen = l

print "avg:",  total * (1.0 / len(df["POLYLINE"]))
print "min:", minLen
print "max:", maxLen