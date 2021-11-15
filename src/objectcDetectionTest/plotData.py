import csv
import matplotlib.pyplot as plt
import sys
import os.path

x = []
y = []

if len(sys.argv) != 3:
  print("usage: graph_data data1.csv data2.csv")
  exit(1)

name, extension = os.path.splitext(sys.argv[1])
if extension != ".csv":
  print("error: files must be .csv")
  exit(1)
with open(sys.argv[1], 'r') as file:
  reader = csv.reader(file, delimiter=',')
  i = 0
  for row in reader:
    x.append(i)
    y.append(float(row[0]))
    i += 1
plt.plot(x, y, '-b', label=name)
x.clear()
y.clear()

name, extension = os.path.splitext(sys.argv[2])
if extension != ".csv":
  print("error: files must be .csv")
  exit(1)
with open(sys.argv[2], 'r') as file:
  reader = csv.reader(file, delimiter=',')
  i = 0
  for row in reader:
    x.append(i)
    y.append(float(row[0]))
    i += 1
plt.plot(x, y, '-r', label=name)
x.clear()
y.clear()

plt.xlabel('Iteraciones')
plt.ylabel('FPS')
plt.title('Evolución de los FPS a lo largo de las iteraciones')
plt.legend(loc="upper left")
plt.show()