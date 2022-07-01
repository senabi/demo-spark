import findspark

findspark.init("/opt/spark")

from pyspark import SparkContext
import findspark
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession
import sys
from preprocess import (
    img_to_array,
    img_to_array_spatial,
    img_to_csv,
    img_to_csv_spatial,
)

spark = SparkSession.builder.appName("Image Segmentation").getOrCreate()

img_path = sys.argv[1]
n_k = int(sys.argv[2])
mode = 1 if sys.argv[3] == "spatial" else 0
if mode == 1:
    path, df, hw = img_to_array_spatial(img_path, spark)
    # path, hw = img_to_csv_spatial(img_path)
else:
    path, df, hw = img_to_array(img_path, spark)
    # path, hw = img_to_csv(img_path)


# df = spark.read.csv(path, header=True, inferSchema=True)
# path = "hdfs://hadoop-master:9000/user/hadoopuser/A.webp-1247x699.csv"
# df = spark.read.csv(path="/tmp/2x2.png-2x2.csv", header=True, inferSchema=True)
df.show()

df.printSchema()

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

input_cols = df.columns
vec_assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
final_data = vec_assembler.transform(df)

kmeans = KMeans(k=n_k, featuresCol="features")
model = kmeans.fit(final_data)
model.transform(final_data).groupBy("prediction").count().show()
predictions = model.transform(final_data)
predictions.show()
predictions = predictions.collect()

centroids = model.clusterCenters()

_width, _height = hw
print(hw)
img = np.zeros((_height, _width, 3))
print(img.shape)
for h in range(_height):
    for w in range(_width):
        img[h, w] = centroids[predictions[_width * h + w]["prediction"]][:3]


# write img
img = Image.fromarray(img.astype("uint8"), "RGB")
result_path = "kmeans-5d-result.png" if mode == 1 else "kmeans-3d-result.png"
img.save(result_path)
