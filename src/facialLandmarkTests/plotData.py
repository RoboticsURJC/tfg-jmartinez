import csv
from distutils import extension
import random as random
import matplotlib.pyplot as plt
import sys
import os.path

x = []
y = []
colors = []

if len(sys.argv) < 2 or len(sys.argv) > 4:
    print("usage: plotData file1.csv [file2.csv ...]")
    exit(1)

for i in range(1, len(sys.argv)):
    name, extension = os.path.splitext(sys.argv[i])
    r = random.random()
    b = random.random()
    g = random.random()
    if extension != ".csv":
        print("error: files must be .csv")
        exit(1)
    with open(sys.argv[i], 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            x.append(float(row[1]))
            y.append(float(row[0]))
        plt.plot(x, y, c=(r,b,g), label=sys.argv[i])
        x.clear()
        y.clear()

plt.xlabel('Tiempo de ejecución en segundos')
plt.ylabel('Media de fotogramas por segundo')
plt.title('Media de fotogramas por segundo a lo largo de la ejecución del programa')
plt.legend(loc="upper left")
plt.show()