import csv
from distutils import extension
import random as random
import matplotlib.pyplot as plt
import sys
import os.path

count = 0

with open(sys.argv[1], 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        value = int(row[0])
        if value == 1:
            count += 1
    
print("Fallos: "+str(count))