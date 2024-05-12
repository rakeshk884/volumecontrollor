import os
import cvzone 
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)  # Initialize video capture
Classifier = Classifier("C:/Users/rakes/PycharmProjects/volumecontrollor/Resources/Model/keras_model.h5", "C:/Users/rakes/PycharmProjects/volumecontrollor/Resources/Model/labels.txt")
imgArrow = cv2.imread("C:/Users/rakes/PycharmProjects/volumecontrollor/Resources/arrow.png",cv2.IMREAD_UNCHANGED)
classIDBin = 0
# Import all the waste Images
imgWasteList = []
pathFolderWaste = "C:/Users/rakes/PycharmProjects/volumecontrollor/Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste,path),cv2.IMREAD_UNCHANGED))
# Import all the waste Images
imgBinsList = []
pathFolderBins = "C:/Users/rakes/PycharmProjects/volumecontrollor/Resources/Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins,path),cv2.IMREAD_UNCHANGED))
# 0 = Recyclable
# 1 = Hazardous
# 2 = Food
# 3 = Residual
classDic = {0:None,
            1:0,
            2:0,
            3:3,
            4:3,
            5:1,
            6:1,
            7:2,
            8:2}

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img,(454,340))
    imgBackground = cv2.imread("C:/Users/rakes/PycharmProjects/volumecontrollor/Resources/background.png")
    prediction = Classifier.getPrediction(img)
    classID = prediction[1]
    print(classID)
    if classID !=0:
      imgBackground = cvzone.overlayPNG(imgBackground,imgWasteList[classID - 1],(950, 127))
      imgBackground = cvzone.overlayPNG(imgBackground,imgArrow,(978, 320))
      classIDBin = classDic[classID]
    imgBackground = cvzone.overlayPNG(imgBackground,imgBinsList[classIDBin],(895, 375))
    imgBackground[148:148+340,159:159+454] = imgResize
    # Displays
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)