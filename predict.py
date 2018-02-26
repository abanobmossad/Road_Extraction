from datetime import datetime
from PIL import Image
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

t_total1 = time.time()
t1 = time.time()
print(str(datetime.now()) + ': initializing input data...')

rectSize = 5

# enter the image pass here
image_path = r'E:\mass_roads\valid\sat\23128930_15.TIFF'

inputImage = Image.open(image_path)
inputImageXSize, inputImageYSize = inputImage.size

outputImage = inputImage.crop(
    (rectSize // 2, rectSize // 2, inputImageXSize - (rectSize // 2), inputImageYSize - (rectSize // 2)))
outputImageXSize, outputImageYSize = outputImage.size

print(str(datetime.now()) + ': initializing model...')
featureColumns = [tf.contrib.layers.real_valued_column("", dimension=79)]

hiddenUnits = [100, 150, 100, 50]

classes = 3

classifier = tf.contrib.learn.DNNClassifier(feature_columns=featureColumns,
                                            hidden_units=hiddenUnits,
                                            n_classes=classes,
                                            model_dir='model')

t2 = time.time()
print("initializing model time :", (t2 - t1)/60)


def extractFeatures():
    features = np.zeros((((inputImageXSize - ((rectSize // 2) * 2)) * (inputImageYSize - ((rectSize // 2) * 2))),
                         rectSize * rectSize * 3), dtype=np.int)
    rowIndex = 0

    for x in range(rectSize // 2, inputImageXSize - (rectSize // 2)):
        for y in range(rectSize // 2, inputImageYSize - (rectSize // 2)):
            rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
            subImage = inputImage.crop(rect).load()
            colIndex = 0
            for i in range(rectSize):
                for j in range(rectSize):
                    features[rowIndex, colIndex] = subImage[i, j][0]
                    colIndex += 1
                    features[rowIndex, colIndex] = subImage[i, j][1]
                    colIndex += 1
                    features[rowIndex, colIndex] = subImage[i, j][2]
                    colIndex += 1

            rowIndex += 1

    return features


def constructOutputImage(predictions):
    outputImagePixels = outputImage.load()
    rowIndex = 0
    for x in range(outputImageXSize):
        for y in range(outputImageYSize):
            if predictions[rowIndex]==1:
                outputImagePixels[x, y] = (255, 255,255)
            else:
                outputImagePixels[x, y] = (0, 0, 0)

            rowIndex += 1


t1 = time.time()
print(str(datetime.now()) + ': processing image')
predictions = list(classifier.predict_classes(input_fn=extractFeatures))
t2 = time.time()
print("Extract features and predicting time :", (t2 - t1)/60)

t1 = time.time()
print(str(datetime.now()) + ': constructing output image...')
constructOutputImage(predictions)
t2 = time.time()

print("constructing output image and ploting time : ", (t2 - t1)/60)
plt.figure()
plt.imshow(outputImage)
plt.show()

print(str(datetime.now()) + ': saving output image...')
outputImage.save('Results/Second-result_road.png', 'JPEG')
t_total2 = time.time()

print(str(datetime.now()) + ': Total time for predicting : ', (t_total2 - t_total1)/60)
