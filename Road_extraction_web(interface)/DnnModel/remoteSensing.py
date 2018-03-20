from datetime import datetime
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class RemoteSensing:

    def __init__(self, image_path, image_name, classifier, SAVED_LOCATION):
        self.image_name = image_name
        self.classifier = classifier
        self.rectSize = 5
        self.SAVED_LOCATION = SAVED_LOCATION
        self.inputImage = Image.open(image_path).resize((1000, 1000), Image.ANTIALIAS)

        self.inputImageXSize, self.inputImageYSize = self.inputImage.size

        self.outputImage = self.inputImage.crop(
            (self.rectSize // 2, self.rectSize // 2, self.inputImageXSize - (self.rectSize // 2),
             self.inputImageYSize - (self.rectSize // 2)))
        self.outputImageXSize, self.outputImageYSize = self.outputImage.size

    def extractFeatures(self):
        features = np.zeros((((self.inputImageXSize - ((self.rectSize // 2) * 2)) * (
                self.inputImageYSize - ((self.rectSize // 2) * 2))),
                             self.rectSize * self.rectSize * 3), dtype=np.int)
        rowIndex = 0

        for x in range(self.rectSize // 2, self.inputImageXSize - (self.rectSize // 2)):
            for y in range(self.rectSize // 2, self.inputImageYSize - (self.rectSize // 2)):
                rect = (x - (self.rectSize // 2), y - (self.rectSize // 2), x + (self.rectSize // 2) + 1,
                        y + (self.rectSize // 2) + 1)
                subImage = self.inputImage.crop(rect).load()

                colIndex = 0
                for i in range(self.rectSize):
                    for j in range(self.rectSize):
                        features[rowIndex, colIndex] = subImage[i, j][0]
                        colIndex += 1
                        features[rowIndex, colIndex] = subImage[i, j][1]
                        colIndex += 1
                        features[rowIndex, colIndex] = subImage[i, j][2]
                        colIndex += 1
                rowIndex += 1

        return features

    def constructOutputImage(self, predictions):
        outputImagePixels = self.outputImage.load()
        rowIndex = 0
        for x in range(self.outputImageXSize):
            for y in range(self.outputImageYSize):
                if predictions[rowIndex] == 1:
                    outputImagePixels[x, y] = (0, 255, 0)
                # elif predictions[rowIndex] == 2:
                #     outputImagePixels[x, y] = (0, 255, 0)
                else:
                    outputImagePixels[x, y] = (0, 0, 0)

                rowIndex += 1

    def aerial_predicting(self):

        print(str(datetime.now()) + "Extract features and predicting time ")
        predictions = list(self.classifier.predict_classes(input_fn=self.extractFeatures))
        print(str(datetime.now()) + ': constructing output image...')
        self.constructOutputImage(predictions)

        final_image = ImageChops.add(self.inputImage, self.outputImage, scale=1.0, offset=0)

        print(str(datetime.now()) + ': saving output image...')
        self.image_name = self.image_name.split('.')
        final_image.save(self.SAVED_LOCATION + self.image_name[0] + '.png', 'JPEG')
        self.outputImage.save(self.SAVED_LOCATION + self.image_name[0] + '.jpg', 'JPEG')
