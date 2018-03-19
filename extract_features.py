import random
from os import listdir
from PIL import Image
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import multiblock_lbp, greycomatrix, greycoprops, CENSURE, local_binary_pattern, ORB
from skimage.transform import integral_image

''' 
    @:param

    [1] this file takes the train images folder path
    [2] this file takes the test images folder path
    [3] this file takes the valid images folder path
    [4] all folders contain (2) sub folders input-images [the satellite images] & output-images [the groundTruth images B&W]
    [5] the input-images and output-images folders must be the same numbers of images 
    [6] all images must be the same size (1500 x 1500)

    @:description 

    [1] the script crop all images into sub-images (5 x 5) for all 3 channels (RGB)
    [2] thin compare each sub-image with the same in the output-images 
    [3] and put 1 if the pixel is Road or 0 if it's not
    [4] and put all in a single line in CSV file with [0 or 1] in the end depending on step [3]
    [5] do this with all the folders mentioned above and gegenerate 3 CSV files [train, test, valid]
    [6] contain all the feature with the right values used to train, test and predict 

'''
# line limit per CSV file
# for training dataset
trainLinesLimit = 1000000
# for testing dataset
testLinesLimit = 800000

# time to check total time to process this images to CSV files
startTotalTime = time.time()

# Train path
trainInputImagesPath = r'E:\Dataset\Train\Train-input'
trainOutputImagesPath = r'E:\Dataset\Train\Train-output-roads'

# Test path

testInputImagesPath = r'E:\Dataset\Test\Test-input'
testOutputImagesPath = r'E:\Dataset\Test\Test-output-roads'

# buildings path
targetBuildingsImagesPath = r'E:\Dataset\Train\Train-output-buildings'
testBuildingsImagesPath = r'E:\Dataset\Test\Test-output-buildings'

trainInputImagesFiles = listdir(trainInputImagesPath)
trainOutputImagesFiles = listdir(trainOutputImagesPath)

testInputImagesFiles = listdir(testInputImagesPath)
testOutputImagesFiles = listdir(testOutputImagesPath)

targetBuildingsImagesFiles = listdir(targetBuildingsImagesPath)
testBuildingsImagesFiles = listdir(testBuildingsImagesPath)
# check if the folders are the same length


print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Train Data Images Files:', len(trainInputImagesFiles))
print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Train Target Images Files:', len(trainOutputImagesFiles))

if (len(trainInputImagesFiles) != len(trainOutputImagesFiles)):
    raise Exception('train input images and output images number mismatch')

print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Test Data Images Files:', len(testInputImagesFiles))
print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Test Target Images Files:', len(testOutputImagesFiles))

print("-------------------------------------------------------------------")

if (len(testInputImagesFiles) != len(testOutputImagesFiles)):
    raise Exception('test input images and output images number mismatch')

for i in range(len(trainInputImagesFiles)):
    inputImageFile = trainInputImagesFiles[i][:-5]
    outputImageFile = trainOutputImagesFiles[i][:-4]
    if (inputImageFile != outputImageFile):
        raise Exception('train inputImageFile and outputImageFile mismatch at index', str(i))

for i in range(len(testInputImagesFiles)):
    inputImageFile = testInputImagesFiles[i][:-5]
    outputImageFile = testOutputImagesFiles[i][:-4]
    if (inputImageFile != outputImageFile):
        raise Exception('test inputImageFile and outputImageFile mismatch at index', str(i))

print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Input and output files check success')


def writeDataFile(inputImagePath, outputImagePath, buildingImagePath, inputImageFiles, outputImageFiles,
                  buildingImageFiles, dataFileName, linesLimit: int):
    dataFile = open(dataFileName, 'w')
    rectSize = 5
    linesCount = 0
    linesLimitPerImage = (linesLimit / len(inputImageFiles)) + 1
    for i in range(len(inputImageFiles)):
        print(str(datetime.now()) + ': Extract features from image', i + 1, "\"", inputImageFiles[i], "\"")
        linesCountPerImage = 0
        inputImage = Image.open(inputImagePath + '/' + inputImageFiles[i])
        inputImageXSize, inputImageYSize = inputImage.size

        outputImage = Image.open(outputImagePath + '/' + outputImageFiles[i])
        outputImageXSize, outputImageYSize = outputImage.size
        outputImagePixels = outputImage.load()

        buildingImage = Image.open(buildingImagePath + '/' + buildingImageFiles[i])
        buildingImage = buildingImage.convert('L')
        buildingImagePixels = buildingImage.load()

        if ((inputImageXSize != outputImageXSize) or (inputImageYSize != outputImageYSize)):
            raise Exception('train inputImage and outputImage mismatch at index', str(i))

        outputImageRoadPixelsArr = []
        outputImageBuildingPixelsArr = []
        outputImageNonRoadPixelsArr = []

        for x in range(rectSize // 2, inputImageXSize - (rectSize // 2)):
            for y in range(rectSize // 2, inputImageYSize - (rectSize // 2)):

                isRoadPixel = outputImagePixels[x, y]
                isBuildingPixel = buildingImagePixels[x, y]
                if (isRoadPixel):
                    outputImageRoadPixelsArr.append((x, y))
                elif (isBuildingPixel):
                    outputImageBuildingPixelsArr.append((x, y))
                else:
                    outputImageNonRoadPixelsArr.append((x, y))

        random.shuffle(outputImageRoadPixelsArr)
        random.shuffle(outputImageBuildingPixelsArr)
        random.shuffle(outputImageNonRoadPixelsArr)

        print("Road Pixies number :", len(outputImageRoadPixelsArr))
        print("Building Pixies number :", len(outputImageBuildingPixelsArr))
        print("Non buildings and roads Pixies numbers :", len(outputImageNonRoadPixelsArr))

        for m in range(len(outputImageRoadPixelsArr)):
            if (linesCountPerImage >= linesLimitPerImage):
                break

            if (((m * 2) + 1) >= len(outputImageNonRoadPixelsArr)):
                break

            x = outputImageRoadPixelsArr[m][0]
            y = outputImageRoadPixelsArr[m][1]

            rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
            subImage = inputImage.crop(rect).load()

            # ---------------------GlCM features----------------------
            ROI_roads = np.asarray(inputImage.crop(rect).getdata()).reshape((5, 5, 3)).astype('uint8')
            ROI_roads = rgb2gray(ROI_roads)
            ROI_roads *= 255.0 / ROI_roads.max()
            ROI_roads = ROI_roads.astype('int').astype('uint8')
            g = greycomatrix(ROI_roads, [1, 2], [0, np.pi / 2], normed=True, symmetric=True)
            GlcmRoads = greycoprops(g, 'contrast')

            # print("Roads\n", GlcmRoads)
            # ------------------------------------------


            line = ''
            count = 0
            for i in range(rectSize):
                for j in range(rectSize):
                    line += str(subImage[i, j][0]) + ','
                    line += str(subImage[i, j][1]) + ','
                    line += str(subImage[i, j][2]) + ','
                    count += 1

            for x in range(GlcmRoads.shape[0]):
                for y in range(GlcmRoads.shape[1]):
                    line += str(GlcmRoads[x, y]) + ','


            line += str(1) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)

            for n in range(2):
                x = outputImageNonRoadPixelsArr[(m * 2) + n][0]
                y = outputImageNonRoadPixelsArr[(m * 2) + n][1]

                rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
                subImage = inputImage.crop(rect).load()

                # --------------------GlCM features-----------------------
                ROI_non = np.asarray(inputImage.crop(rect).getdata()).reshape((5, 5, 3)).astype('uint8')
                ROI_non = rgb2gray(ROI_non)
                ROI_non *= 255.0 / ROI_non.max()
                ROI_non = ROI_non.astype('int').astype('uint8')
                g = greycomatrix(ROI_non, [1, 2], [0, np.pi / 2], normed=True, symmetric=True)
                GlcmNon = greycoprops(g, 'contrast')

                # print("NON\n", GlcmNon)
                # ------------------------------------------

                line = ''
                for i in range(rectSize):
                    for j in range(rectSize):
                        line += str(subImage[i, j][0]) + ','
                        line += str(subImage[i, j][1]) + ','
                        line += str(subImage[i, j][2]) + ','

                for x in range(GlcmNon.shape[0]):
                    for y in range(GlcmNon.shape[1]):
                        line += str(GlcmNon[x, y]) + ','

                line += str(0) + '\n'
                linesCount += 1
                linesCountPerImage += 1
                dataFile.write(line)

            x = outputImageBuildingPixelsArr[(m * 2) + n][0]
            y = outputImageBuildingPixelsArr[(m * 2) + n][1]

            rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
            subImage = inputImage.crop(rect).load()

            # -----------------GlCM features---------------------
            ROI_build = np.asarray(inputImage.crop(rect).getdata()).reshape((5, 5, 3)).astype('uint8')
            ROI_build = rgb2gray(ROI_build)
            ROI_build *= 255.0 / ROI_build.max()
            ROI_build = ROI_build.astype('int').astype('uint8')
            g = greycomatrix(ROI_build, [1, 2], [0, np.pi / 2], normed=True, symmetric=True)
            GlcmBuild = greycoprops(g, 'contrast')

            # print("Building\n", GlcmBuild)
            # ------------------------------------------

            line = ''
            for i in range(rectSize):
                for j in range(rectSize):
                    line += str(subImage[i, j][0]) + ','
                    line += str(subImage[i, j][1]) + ','
                    line += str(subImage[i, j][2]) + ','

            for x in range(GlcmBuild.shape[0]):
                for y in range(GlcmBuild.shape[1]):
                    line += str(GlcmBuild[x, y]) + ','

            line += str(2) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)
        print("-------------------------------------------------------------------")
    print(str(datetime.now()) + ': ' + dataFileName + ' linesCount:', linesCount)


trainDataFileName = 'airs-dataset/Train.csv'
testDataFileName = 'airs-dataset/Test.csv'

print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': writing train Data File in CVS file with ', trainLinesLimit,
      " line", "and 10310 line per image")

writeDataFile(trainInputImagesPath, trainOutputImagesPath, targetBuildingsImagesPath, trainInputImagesFiles,
              trainOutputImagesFiles, targetBuildingsImagesFiles, trainDataFileName, trainLinesLimit)
print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Train data CSV file complete.')

print("-------------------------------------------------------------------")
print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': writing test Data File in CVS file with ', testLinesLimit,
      " line")

writeDataFile(testInputImagesPath, testOutputImagesPath, testBuildingsImagesPath, testInputImagesFiles,
              testOutputImagesFiles, testBuildingsImagesFiles, testDataFileName, testLinesLimit)
print('{0:%Y-%m-%d %H:%M}'.format(datetime.now()) + ': Test data CSV file complete.')

# time to check total time to process this images to CSV files
endTotalTime = time.time()

print("Total time the process takes : ", (endTotalTime - startTotalTime) / 60, " Minutes")
