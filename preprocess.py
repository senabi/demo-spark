from PIL import Image
import sys
import numpy as np
import csv
from pyspark.sql import Row


def img_to_csv_spatial(path):
    data = np.array(Image.open(path))
    h, w = data.shape[:2]
    with open(f"{path}-{w}x{h}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "g", "b", "x", "y"])
        for row_ix, row in enumerate(data):
            for pixel_ix, pixel in enumerate(row):
                r, g, b = pixel
                writer.writerow([r, g, b, pixel_ix, row_ix])
    return f"{path}-{w}x{h}.csv", (w, h)


def img_to_csv(path):
    data = np.array(Image.open(path))
    h, w = data.shape[:2]
    with open(f"{path}-{w}x{h}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "g", "b"])
        for row in data:
            for pixel in row:
                writer.writerow(pixel)
    return f"{path}-{w}x{h}.csv", (w, h)


def img_to_array(path, spark):
    data = np.array(Image.open(path))
    h, w = data.shape[:2]
    arr = []
    for row_ix, row in enumerate(data):
        for pixel_ix, pixel in enumerate(row):
            r, g, b = pixel
            arr.append({"r": int(r), "g": int(g), "b": int(b)})
    return path, spark.createDataFrame(arr), (w, h)


def img_to_array_spatial(path, spark):
    data = np.array(Image.open(path))
    h, w = data.shape[:2]
    arr = []
    for row_ix, row in enumerate(data):
        for pixel_ix, pixel in enumerate(row):
            r, g, b = pixel
            arr.append(
                {
                    "r": int(r),
                    "g": int(g),
                    "b": int(b),
                    "x": int(pixel_ix),
                    "y": int(row_ix),
                }
            )
    return path, spark.createDataFrame(arr), (w, h)


if __name__ == "__main__":
    import sys

    img_to_csv(sys.argv[1])
