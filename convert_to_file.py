import random
from os import listdir
from PIL import Image
from datetime import datetime

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

# Train path
trainInputImagesPath = 'E:/mass_roads/Train/Train-input'
trainOutputImagesPath = 'E:/mass_roads/Train/Train-output'

# Test path

testInputImagesPath = 'E:/mass_roads/Test/Test-input'
testOutputImagesPath = 'E:/mass_roads/Test/Test-output'

# validation path

validInputImagesPath = 'E:/mass_roads/Validation/Valid-input'
validOutputImagesPath = 'E:/mass_roads/Validation/Valid-output'

trainInputImagesFiles = listdir(trainInputImagesPath)
trainOutputImagesFiles = listdir(trainOutputImagesPath)

testInputImagesFiles = listdir(testInputImagesPath)
testOutputImagesFiles = listdir(testOutputImagesPath)

validInputImagesFiles = listdir(validInputImagesPath)
validOutputImagesFiles = listdir(validOutputImagesPath)

# check if the folders are the same length

print(str(datetime.now()) + ': trainInputImagesFiles:', len(trainInputImagesFiles))
print(str(datetime.now()) + ': trainOutputImagesFiles:', len(trainOutputImagesFiles))
if (len(trainInputImagesFiles) != len(trainOutputImagesFiles)):
    raise Exception('train input images and output images number mismatch')

print(str(datetime.now()) + ': testInputImagesFiles:', len(testInputImagesFiles))
print(str(datetime.now()) + ': testOutputImagesFiles:', len(testOutputImagesFiles))
if (len(testInputImagesFiles) != len(testOutputImagesFiles)):
    raise Exception('test input images and output images number mismatch')

print(str(datetime.now()) + ': validInputImagesFiles:', len(validInputImagesFiles))
print(str(datetime.now()) + ': validOutputImagesFiles:', len(validOutputImagesFiles))
if (len(validInputImagesFiles) != len(validOutputImagesFiles)):
    raise Exception('valid input images and output images number mismatch')

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

for i in range(len(validInputImagesFiles)):
    inputImageFile = validInputImagesFiles[i][:-5]
    outputImageFile = validOutputImagesFiles[i][:-4]
    if (inputImageFile != outputImageFile):
        raise Exception('valid inputImageFile and outputImageFile mismatch at index', str(i))

print(str(datetime.now()) + ': input and output files check success')


# unused test method
def writeDataFileOld(inputImagePath, outputImagePath, inputImageFiles, outputImageFiles, dataFileName):
    dataFile = open(dataFileName, 'w')
    roadPixel = 1
    nonroadPixel = 0
    neededPixel = 0

    rectSize = 5

    for i in range(len(inputImageFiles)):
        # if(i > 0):
        #    break

        print(str(datetime.now()) + ': prcessing image', i)
        inputImage = Image.open(inputImagePath + '/' + inputImageFiles[i])
        inputImageXSize, inputImageYSize = inputImage.size
        inputImagePixels = inputImage.load()

        outputImage = Image.open(outputImagePath + '/' + outputImageFiles[i])
        outputImageXSize, outputImageYSize = outputImage.size
        outputImagePixels = outputImage.load()

        if ((inputImageXSize != outputImageXSize) or (inputImageYSize != outputImageYSize)):
            raise Exception('train inputImage and outputImage mismatch at index', str(i))

        for x in range(rectSize // 2, inputImageXSize - (rectSize // 2)):
            for y in range(rectSize // 2, inputImageYSize - (rectSize // 2)):
                isRoadPixel = outputImagePixels[x, y]
                if ((isRoadPixel) and (neededPixel != roadPixel)):
                    continue

                if ((not (isRoadPixel)) and (neededPixel == roadPixel)):
                    continue

                neededPixel = ((neededPixel + 1) % 3)
                rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
                subImage = inputImage.crop(rect).load()
                line = ''
                for i in range(rectSize):
                    for j in range(rectSize):
                        line += str(subImage[i, j][0]) + ','
                        line += str(subImage[i, j][1]) + ','
                        line += str(subImage[i, j][2]) + ','

                line += str(roadPixel if isRoadPixel else nonroadPixel) + '\n'
                dataFile.write(line)


def writeDataFile(inputImagePath, outputImagePath, inputImageFiles, outputImageFiles, dataFileName):
    dataFile = open(dataFileName, 'w')
    rectSize = 5
    linesCount = 0
    linesLimit = 200000
    linesLimitPerImage = (linesLimit / len(inputImageFiles)) + 1

    for i in range(len(inputImageFiles)):
        print(str(datetime.now()) + ': prcessing image', i + 1)
        linesCountPerImage = 0
        inputImage = Image.open(inputImagePath + '/' + inputImageFiles[i])
        inputImageXSize, inputImageYSize = inputImage.size
        # inputImagePixels = inputImage.load()

        outputImage = Image.open(outputImagePath + '/' + outputImageFiles[i])
        outputImageXSize, outputImageYSize = outputImage.size
        outputImagePixels = outputImage.load()

        if ((inputImageXSize != outputImageXSize) or (inputImageYSize != outputImageYSize)):
            raise Exception('train inputImage and outputImage mismatch at index', str(i))

        outputImageRoadPixelsArr = []
        outputImageNonRoadPixelsArr = []

        for x in range(rectSize // 2, inputImageXSize - (rectSize // 2)):
            for y in range(rectSize // 2, inputImageYSize - (rectSize // 2)):
                isRoadPixel = outputImagePixels[x, y]
                if (isRoadPixel):
                    outputImageRoadPixelsArr.append((x, y))
                else:
                    outputImageNonRoadPixelsArr.append((x, y))

        random.shuffle(outputImageRoadPixelsArr)
        random.shuffle(outputImageNonRoadPixelsArr)

        for m in range(len(outputImageRoadPixelsArr)):
            if (linesCountPerImage >= linesLimitPerImage):
                break

            if (((m * 2) + 1) >= len(outputImageNonRoadPixelsArr)):
                break

            x = outputImageRoadPixelsArr[m][0]
            y = outputImageRoadPixelsArr[m][1]

            rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
            subImage = inputImage.crop(rect).load()
            line = ''
            count = 0
            for i in range(rectSize):
                for j in range(rectSize):
                    line += str(subImage[i, j][0]) + ','
                    line += str(subImage[i, j][1]) + ','
                    line += str(subImage[i, j][2]) + ','
                    count += 1

            line += str(1) + '\n'
            linesCount += 1
            linesCountPerImage += 1
            dataFile.write(line)

            for n in range(2):
                x = outputImageNonRoadPixelsArr[(m * 2) + n][0]
                y = outputImageNonRoadPixelsArr[(m * 2) + n][1]

                rect = (x - (rectSize // 2), y - (rectSize // 2), x + (rectSize // 2) + 1, y + (rectSize // 2) + 1)
                subImage = inputImage.crop(rect).load()
                line = ''
                for i in range(rectSize):
                    for j in range(rectSize):
                        line += str(subImage[i, j][0]) + ','
                        line += str(subImage[i, j][1]) + ','
                        line += str(subImage[i, j][2]) + ','

                line += str(0) + '\n'
                linesCount += 1
                linesCountPerImage += 1
                dataFile.write(line)

    print(str(datetime.now()) + ': ' + dataFileName + ' linesCount:', linesCount)


trainDataFileName = 'airs-dataset/train.csv'
testDataFileName = 'airs-dataset/test.csv'
validDataFileName = 'airs-dataset/valid.csv'

print(str(datetime.now()) + ': writing trainDataFile')
writeDataFile(trainInputImagesPath, trainOutputImagesPath, trainInputImagesFiles, trainOutputImagesFiles,
              trainDataFileName)
print(str(datetime.now()) + ': trainDataFile complete')

print(str(datetime.now()) + ': writing testDataFile')
writeDataFile(testInputImagesPath, testOutputImagesPath, testInputImagesFiles, testOutputImagesFiles, testDataFileName)
print(str(datetime.now()) + ': testDataFile complete')

print(str(datetime.now()) + ': writing validDataFile')
writeDataFile(validInputImagesPath, validOutputImagesPath, validInputImagesFiles, validOutputImagesFiles,
              validDataFileName)
print(str(datetime.now()) + ': validDataFile complete')
