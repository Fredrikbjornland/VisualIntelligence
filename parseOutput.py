import cv2
import os
from tqdm import tqdm

pathToPredictions = "yolov5/runs/detect/yolo_road_damage7/labels"

def getPredictions(fileName):
    with open(fileName, "r") as file:
        prediction_list = file.read().split("\n")[:-1]
        return prediction_list


def getScreenSizeOfFile(jpg_file):
    im = cv2.imread(jpg_file)
    return im.shape

file_name = "normalizedPredictions.txt"



def normalizePrediction(height, width, label):
    normalizedValues = label[1] * width, label[2] * height, label[3] * width, label[4] * height
    return [str(int(x)) for x in normalizedValues]


def writePrediction(height, width, imageName, labels):
    answer = imageName + ", "
    for label in labels:
        answer += str(int(label[0]) + 1)  + " "
        newLabel = [float(x) for x in label.split(" ")]
        nPred = normalizePrediction(height, width, newLabel)
        for val in nPred:
            answer += val + " "
    answer += "\n"
    with open(file_name, 'a') as f:
        f.write(answer)

def clearOutputFile():
    with open(file_name, 'w') as f:
        pass


def normalzeAllImages():
    clearOutputFile()
    images = [os.path.join("Norway/test/images", x) for x in os.listdir("Norway/test/images") if x[-3:] == "jpg"]
    predictions = [os.path.join(pathToPredictions, x) for x in os.listdir(pathToPredictions) if x[-3:] == "txt"]
    images.sort()
    predictions.sort()
    # Convert and save the annotations
    j = 0
    for i in range(len(images)):
        image_number = images[i][-10:-4]
        prediction_number = predictions[j][-10:-4]
        if image_number == prediction_number:
            image_size = getScreenSizeOfFile(images[i])
            height, width, depth = image_size

            labels = getPredictions(predictions[j])
            writePrediction(height, width, images[i][-17:], labels)
            j += 1
        else:
            emptyAnswer = images[i][-17:] + ",\n"
            with open(file_name, 'a') as f:
                f.write(emptyAnswer)


normalzeAllImages()