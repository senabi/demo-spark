# from lib2to3.pgen2.literals import simple_escapes
import sys
from turtle import width
from pyspark import SparkContext
import findspark
import pyspark
import random
import numpy as np
from PIL import Image

# import matplotlib.pyplot as plt

findspark.init("/opt/spark")

conf = pyspark.SparkConf("spark://hadoop-master:7077")
sc = SparkContext(conf=conf)

path_img = sys.argv[1]
n_centroids = int(sys.argv[2])
n_iterations = int(sys.argv[3])

img = np.asarray(Image.open(path_img))
_height, _width, _ = img.shape
print(img.shape)
print("Image: ", path_img)
print(img)

# centroids = np.zeros((n_centroids, 3))  # [[rgb] [rgb] [rgb]]
centroids = np.random.randint(0, 255, (n_centroids, 3), dtype=np.uint8)
print("Centroids: ", type(centroids))
# print(centroids[0])
imgl = []

# ALAMACCENAR EL TAMANIO DE NUESTRA IAMGE  WIDTH HEGIH
# imgl
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        imgl.append([img[i, j], random.randint(0, n_centroids - 1), centroids])


print("imgl: ")
print(imgl)

# arr.reshape(-1, arr.shape[-1])

# data = [[c][rgb][l]]


class functor(object):
    def __init__(self, centroids):
        self.centroids = centroids

    def nearest(self, data):
        smallest = [10000000, None]
        data[2] = self.centroids
        for ix, c in enumerate(data[2]):
            dist = np.linalg.norm(data[1] - c)
            if dist < smallest[0]:
                smallest = [dist, ix]
        data[1] = smallest[1]
        return data


def nearest(data):
    smallest = [10000000, None]
    for ix, c in enumerate(data[2]):
        print(c)
        dist = np.linalg.norm(data[1] - c)
        if dist < smallest[0]:
            smallest = [dist, ix]
    data[1] = smallest[1]
    print(data[2])
    return data


print("==centroids: ")
print(centroids)

dfi = sc.parallelize(imgl)  # .

for i in range(0, n_iterations):  # .
    classFunctor = functor(centroids)
    dfi = dfi.map(classFunctor.nearest)  # .
    data_list = dfi.take(len(imgl))

    temp_centroides = centroids  # ?  temp_centroides = [ [0,0,0] [0,0,0] ]
    temp_cantidad = np.ones(len(centroids), dtype=np.uint8)  # .
    # print("temp_centroides: ", temp_centroides)

    print("==centroides it:", i)
    print(centroids)

    print("data_list: ")
    print(data_list)

    # print(type(data))
    for j in range(len(imgl)):  # .
        temp_centroides[data_list[j][1]] += data_list[j][0]  # ?  [rgb] += [rgb]
        temp_cantidad[data_list[j][1]] += 1  #

    print("temp_centroides: ", temp_centroides.dtype)
    print("temp_cantidad: ", temp_cantidad.dtype)

    for j in range(len(centroids)):
        temp_centroides[j] = temp_centroides[j] / temp_cantidad[j]  # ? [rgb] /= 3 ?
    centroids = temp_centroides


simg = np.zeros(img.shape)

for h in range(_height):
    for w in range(_width):
        print("i: ", h, "j: ", w)
        simg[h, w] = centroids[imgl[_width * h + w][1]]

print("simg: ")
print(simg)
print(simg.shape)

# write numpy array to image
img = Image.fromarray(simg.astype("uint8"), "RGB")
# # write image to disk
img.save("kmeans.png")

sc.stop()

# plt.imshow(simg)
# plt.show()
