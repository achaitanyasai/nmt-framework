import csv
import sys

sourceFile = sys.argv[1]
sf = open(sourceFile)

targetFile = sys.argv[2]
tf = open(targetFile)

outputFile = sys.argv[3]
of = open(outputFile, 'w')
writer = csv.writer(of, delimiter=",", quoting=csv.QUOTE_MINIMAL)

for sourceLine, targetLine in zip(sf.readlines(), tf.readlines()):
    sourceLine = sourceLine.lstrip().rstrip().lower()
    targetLine = targetLine.lstrip().rstrip().lower()
    writer.writerow([sourceLine, targetLine])

sf.close()
tf.close()
of.close()
