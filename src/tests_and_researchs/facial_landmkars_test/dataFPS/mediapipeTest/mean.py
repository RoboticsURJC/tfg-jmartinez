import csv
from distutils import extension
import random as random
import matplotlib.pyplot as plt
import sys
import os.path

sum = 0
count = 0

with open(sys.argv[1], 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        value = float(row[0])
        if value > 0:
            sum += value
            count += 1
    
mean = sum/count
print("Media: "+str(mean))