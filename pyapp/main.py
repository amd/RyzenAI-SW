##########################################################
#  Build Event UI Demo Application                       #
#                                                        #
##########################################################
import sys
import numpy as np
import os
import shutil
import json
import math
import threading
import requests
from PySide6 import QtCore, QtGui, QtQml
from PySide6.QtGui import QGuiApplication,  QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QDialog, QFileDialog, QWidget
from PySide6.QtCore import QFile, Signal, QDir, QUrl, Property
from PySide6 import QtCore
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QTimer, QObject, Signal, QFile, Signal, QDir, Slot, QThread, QRunnable, QThreadPool


#the ML detection library
sys.path.insert(1, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.insert(1, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "models", "yolov5")))
sys.path.insert(1, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "models", "retinaface")))
import yolov5.yolov5_detect
import retinaface.retinaface_detect

class HistogramListModal(QObject):
    modelChanged = Signal()

    def __init__(self, parent=None):
        super(HistogramListModal, self).__init__(parent)
        self.propertyList = []
    
    def model(self):
        return self.propertyList

    def appendToModel(self, histmodel):
        self.propertyList.append(histmodel)
        print("listcount:", len(self.propertyList))
        self.modelChanged.emit()

    def clearModel(self):
        self.propertyList.clear()
        self.modelChanged.emit()

    model = Property("QVariantList", fget=model, notify=modelChanged)


class HistogramModal(QObject):

    def __init__(self, Category="", Count=0 , FolderPath="", BarLength=1.0):
        QObject.__init__(self)
        self.Category_ = Category
        self.Count_ = Count
        self.FolderPath_ = FolderPath
        self.BarLength_ = BarLength

    @Property(int)
    def Count(self):
        return self.Count_

    @Count.setter
    def setCount(self, value):
        self.Count_ = value

    @Property(str)
    def Category(self):
        return self.Category_

    @Category.setter
    def setCategory(self, value):
        self.Category = value

    @Property(str)
    def FolderPath(self):
        return self.FolderPath_

    @FolderPath.setter
    def setFolderPath(self, value):
        self.FolderPath_ = value

    @Property(float)
    def BarLength(self):
        return self.BarLength_



class Backend(QObject):
    selectedFolderPath = Signal(str)
    scanProcessingCompleted = Signal()
    scanWithoutIPUProcessingCompleted = Signal()
    totalCountChanged = Signal()
    categoryMaxBarCountChanged = Signal()
    currentProcessedCountChanged = Signal()
    currentProcessedCountFaceDetectChanged = Signal()
    faceDetectMaxBarCountChanged = Signal()
    showWarningMessage = Signal()
    previousExceptionMessageChanged = Signal()

    def __init__(self, histListModalData, histListModalDataFaceDetect):
        QObject.__init__(self)
        self.histListModal_ = histListModalData
        self.histListModalFaceDetect_ = histListModalDataFaceDetect
        self.currentProcessedCount_ = 0  # TODO: update this value
        self.currentProcessedFaceDetectCount_ = 0
        self.totalCount_ = 0  # TODO: update this value
        self.categoryMaxBarCount_ = 0
        self.faceDetectMaxBarCount_ = 0
        self.previousExceptionMessage_ = ""

    def getCurrentProcessedCount(self):
        return self.currentProcessedCount_

    def setCurrentProcessedCount(self, value):
        self.currentProcessedCount_ = value
        self.currentProcessedCountChanged.emit()

    def getCurrentProcessedCountFaceDetect(self):
        return self.currentProcessedFaceDetectCount_

    def setCurrentProcessedCountFaceDetect(self, value):
        self.currentProcessedFaceDetectCount_ = value
        self.currentProcessedCountFaceDetectChanged.emit()

    def getTotalCount(self):
        return self.totalCount_

    def setTotalCount(self, value):
        self.totalCount_ = value
        self.totalCountChanged.emit()

    def getCategoryMaxBarCount(self):
        return self.categoryMaxBarCount_

    def setCategoryMaxBarCount(self, value):
        self.categoryMaxBarCount_ = value
        self.categoryMaxBarCountChanged.emit()


    def getFaceDetectMaxBarCount(self):
        return self.faceDetectMaxBarCount_

    def setFaceDetectMaxBarCount(self, value):
        self.faceDetectMaxBarCount_ = value
        self.faceDetectMaxBarCountChanged.emit()

    def getpreviousExceptionMessage(self):
        return self.previousExceptionMessage_

    def setpreviousExceptionMessage(self, value):
        self.previousExceptionMessage_ = value
        self.previousExceptionMessageChanged.emit()

    currentProcessedCount = Property(int, getCurrentProcessedCount, setCurrentProcessedCount, notify=currentProcessedCountChanged)
    currentProcessedFaceDetectCount = Property(int, getCurrentProcessedCountFaceDetect, setCurrentProcessedCountFaceDetect, notify=currentProcessedCountFaceDetectChanged)
    totalCount = Property(int, getTotalCount, setTotalCount, notify=totalCountChanged)
    categoryMaxBarCount = Property(int, getCategoryMaxBarCount, setCategoryMaxBarCount, notify=categoryMaxBarCountChanged)
    faceDetectMaxBarCount = Property(int, getFaceDetectMaxBarCount, setFaceDetectMaxBarCount, notify=faceDetectMaxBarCountChanged)
    previousExceptionMessage = Property(str, getpreviousExceptionMessage, setpreviousExceptionMessage, notify=previousExceptionMessageChanged)

    shouldAbort = False
    runningThreads = list()

    class performImageDetection(QRunnable):
        def __init__(self, backend, inputFolderPath, detetImagesAlgroBackend='cpu'):
            QRunnable.__init__(self)
            self.backend = backend
            self.inputFolderPath = inputFolderPath
            self.detetImagesAlgroBackend = detetImagesAlgroBackend

        def run2(self, backend, inputFolderPath):
            backend.runningThreads.append(1)
            #print("IN RUNNABLE RUN2")
            rootDirectory = os.path.dirname(sys.executable)
            if os.path.basename(sys.executable).casefold() == "python.exe".casefold():
                rootDirectory = os.path.dirname(__file__) #launched by: python.exe main.py, so use directory of where main.py is located

            outputDirectory = os.path.join(rootDirectory, "Output", "ObjectDetect")
            outputJsonLocation = os.path.join(rootDirectory, "ObjectDetectOutput.json")

            if os.path.exists(outputDirectory):
                shutil.rmtree(outputDirectory)

            #comment out these for testing so we don't have to re-run the detection script
            if os.path.exists(outputJsonLocation):
                os.remove(outputJsonLocation)

            os.makedirs(outputDirectory)

            detectDataFromJson = ""

            backend.setCurrentProcessedCount(0)
            def updateProgressCallback(result):
                if(backend.shouldAbort):
                    raise GeneratorExit
                #currentProgress = backend.getCurrentProcessedCount();
                #print("@@@@ CURRENT PROCESS = {0}", currentProgress)
                backend.setCurrentProcessedCount(backend.getCurrentProcessedCount() + 1)

            # run the detection script
            if True:
                #print("Test")
                #weightspath = os.path.join(rootDirectory, "yolov5_detect", "model", "yolov5s6.onnx")
                weightspath = os.path.join(os.path.dirname(__file__), "..", "models", "yolov5", "model", "yolov5s6.onnx")
                print(weightspath)
                try:
                    runner = yolov5.yolov5_detect.Runner(weights=weightspath, ep=self.detetImagesAlgroBackend)
                    json_str = runner.run(path=inputFolderPath, callback=updateProgressCallback)
                except GeneratorExit:
                    print("inside GeneratorExit yolov5")
                    return
                except Exception as inst:
                    print("inside exception")
                    print(inst)
                    backend.performStopScan()
                    backend.setpreviousExceptionMessage(str(inst))
                    backend.showWarningMessage.emit()
                    return
                except:
                    print("unknown exception")
                    backend.performStopScan()
                    backend.setpreviousExceptionMessage("Unknown runtime exception when running yolov5")
                    backend.showWarningMessage.emit()
                    return

                json_file = open(outputJsonLocation, "w")
                json_file.write(json_str)
                json_file.close()
            else:
                batFileToRun = os.path.join(rootDirectory, "bin", "run.bat")
                myprocess = subprocess.Popen([batFileToRun, inputFolderPath, outputJsonLocation], cwd=os.path.dirname(batFileToRun), shell=True)
                myprocess.communicate() #wait until done.
                #process the data from the .json output
                print(outputJsonLocation)

            detectJSONfileObject = open(outputJsonLocation, "r")
            detectDataFromJson = json.loads(detectJSONfileObject.read())
            detectJSONfileObject.close()

            detectedObjectsDictionary = {}
            class DetectedObjectsStruct:
                def __init__(self, objectname, folderPath, count = 0):
                    self.objectname = objectname
                    self.count = count
                    self.folderPath = folderPath

            for result in detectDataFromJson["results"]:
                imageFileName = os.path.basename(os.path.realpath(result["image"]))
                #print(imageFileName)
                objectsDetected = set()
                for detectedobject in result["objects"]:
                    label = detectedobject["label"]
                    #print(label)
                    if not (label in objectsDetected):
                        objectsDetected.add(label)
                        itemLinkDirectory = os.path.join(outputDirectory, label)
                        itemLinkFullPath = os.path.join(itemLinkDirectory,imageFileName) 
                        if not os.path.exists(itemLinkDirectory):
                            os.makedirs(itemLinkDirectory)
                        os.link(os.path.join(inputFolderPath, imageFileName), itemLinkFullPath)
                        #symlink requires to be run in admin mode
                        #os.symlink(os.path.join(inputFolderPath, imageFileName), itemLinkFullPath)

                        if not (label in detectedObjectsDictionary):
                            detectedObject = DetectedObjectsStruct(label, itemLinkDirectory)
                            detectedObjectsDictionary[label] = detectedObject
                        detectedObjectsDictionary[label].count += 1

            #for detectedObject in detectedObjectsDictionary:
            #    print("{0}- Count: {1} \t- {2}".format(detectedObjectsDictionary[detectedObject].objectname.ljust(30), detectedObjectsDictionary[detectedObject].count, detectedObjectsDictionary[detectedObject].folderPath ))


            #print("sort alphabetically")
            #detectedObjectsListSortAlpha = sorted(detectedObjectsDictionary.values(), key=lambda x: x.objectname)
            #for detectedObject in detectedObjectsListSortAlpha:
            #    print("{0}- Count: {1} \t- {2}".format(detectedObject.objectname.ljust(30), detectedObject.count, detectedObject.folderPath ))

            print("sort by count")
            detectedObjectsListSortCount = sorted(detectedObjectsDictionary.values(), key=lambda x: (x.count * -1, x.objectname), reverse = False)
            if len(detectedObjectsDictionary) == 0:
                backend.setCategoryMaxBarCount(0)
            else:
                maxCount = detectedObjectsListSortCount[0].count
                maxValueToShowOnChart = 5 * math.ceil(maxCount/5)
                backend.setCategoryMaxBarCount(maxValueToShowOnChart)
                for detectedObject in detectedObjectsListSortCount:
                    print("{0}- Count: {1} \t- {2}".format(detectedObject.objectname.ljust(30), detectedObject.count, detectedObject.folderPath ))
                    obj = HistogramModal(detectedObject.objectname, detectedObject.count, detectedObject.folderPath, detectedObject.count/maxValueToShowOnChart)
                    backend.histListModal_.appendToModel(obj)
            
            backend.runningThreads.pop()
            if not (bool(backend.runningThreads)):
                if(self.detetImagesAlgroBackend == 'azure') :
                    backend.scanWithoutIPUProcessingCompleted.emit()
                if(self.detetImagesAlgroBackend == 'ipu'):
                    backend.scanProcessingCompleted.emit()

        def run(self):
            #print("IN RUNNABLE RUN")
            self.run2(self.backend, self.inputFolderPath)
    #end performImageDetection


    class performFaceDetection(QRunnable):
        def __init__(self, backend, inputFolderPath, detectFacesAlgroBackend='cpu'):
            QRunnable.__init__(self)
            self.backend = backend
            self.inputFolderPath = inputFolderPath
            self.detectFacesAlgroBackend = detectFacesAlgroBackend

        def run2(self, backend, inputFolderPath):
            backend.runningThreads.append(2)
            #print("IN RUNNABLE RUN2")
            rootDirectory = os.path.dirname(sys.executable)
            if os.path.basename(sys.executable).casefold() == "python.exe".casefold():
                rootDirectory = os.path.dirname(__file__) #launched by: python.exe main.py, so use directory of where main.py is located

            outputDirectory = os.path.join(rootDirectory, "Output", "FaceDetect")
            outputJsonLocation = os.path.join(rootDirectory, "FaceDetectOutput.json")

            if os.path.exists(outputDirectory):
                shutil.rmtree(outputDirectory)

            #comment out these for testing so we don't have to re-run the detection script
            if os.path.exists(outputJsonLocation):
                os.remove(outputJsonLocation)

            os.makedirs(outputDirectory)

            detectDataFromJson = ""

            backend.setCurrentProcessedCountFaceDetect(0)
            def updateProgressCallback(result):
                if(backend.shouldAbort):
                    raise GeneratorExit
                #currentProgress = backend.getCurrentProcessedCount();
                #print("@@@@ CURRENT PROCESS = {0}", currentProgress)
                backend.setCurrentProcessedCountFaceDetect(backend.getCurrentProcessedCountFaceDetect() + 1)

            # run the detection script
            if True:
                #print("Test")
                weightspath = os.path.join(os.path.dirname(__file__), "..", "models", "retinaface", "model", "RetinaFace_int.onnx")
                print(weightspath)
                try:
                    runner = retinaface.retinaface_detect.Runner(weights=weightspath, ep=self.detectFacesAlgroBackend)
                    json_str = runner.run(path=inputFolderPath, callback=updateProgressCallback)
                except GeneratorExit:
                    print("inside GeneratorExit retinaface")
                    return
                except Exception as inst:
                    print("inside exception")
                    print(inst)
                    backend.performStopScan()
                    backend.setpreviousExceptionMessage(str(inst))
                    backend.showWarningMessage.emit()
                    return
                except:
                    print("unknown exception")
                    backend.performStopScan()
                    backend.setpreviousExceptionMessage("Unknown runtime exception when running retinaface")
                    backend.showWarningMessage.emit()
                    return
                json_file = open(outputJsonLocation, "w")
                json_file.write(json_str)
                json_file.close()
            else:
                batFileToRun = os.path.join(rootDirectory, "bin", "run.bat")
                myprocess = subprocess.Popen([batFileToRun, inputFolderPath, outputJsonLocation], cwd=os.path.dirname(batFileToRun), shell=True)
                myprocess.communicate() #wait until done.
                #process the data from the .json output
                print(outputJsonLocation)

            detectJSONfileObject = open(outputJsonLocation, "r")
            detectDataFromJson = json.loads(detectJSONfileObject.read())
            detectJSONfileObject.close()

            detectedObjectsDictionary = {}
            class DetectedObjectsStruct:
                def __init__(self, numFaces, folderPath, count = 0):
                    self.numFaces = numFaces
                    self.count = count
                    self.folderPath = folderPath

            for result in detectDataFromJson["results"]:
                imageFileName = os.path.basename(os.path.realpath(result["image"]))
                #print(imageFileName)
                numDetectedFaces = len(result["faces"])
                itemLinkDirectory = os.path.join(outputDirectory, str(numDetectedFaces))
                itemLinkFullPath = os.path.join(itemLinkDirectory,imageFileName)
                #print("{0} - {1}".format(imageFileName, numDetectedFaces))
                if not (numDetectedFaces in detectedObjectsDictionary):
                    detectedObject = DetectedObjectsStruct(numDetectedFaces, itemLinkDirectory)
                    detectedObjectsDictionary[numDetectedFaces] = detectedObject
                    os.makedirs(itemLinkDirectory)
                detectedObjectsDictionary[numDetectedFaces].count += 1
                os.link(os.path.join(inputFolderPath, imageFileName), itemLinkFullPath)
                #symlink requires to be run in admin mode
                #os.symlink(os.path.join(inputFolderPath, imageFileName), itemLinkFullPath)
            
            detectedObjectsDictionaryKeys = sorted(detectedObjectsDictionary)
            if len(detectedObjectsDictionaryKeys) == 0:
                backend.setFaceDetectMaxBarCount(0)
            else:
                maxNumFaces = detectedObjectsDictionaryKeys[-1]
                maxNumFacesCount = 0
                for key in detectedObjectsDictionaryKeys:
                    if maxNumFacesCount < detectedObjectsDictionary[key].count:
                        maxNumFacesCount = detectedObjectsDictionary[key].count

                maxValueToShowOnChart = 5 * math.ceil(maxNumFacesCount/5)
                backend.setFaceDetectMaxBarCount(maxValueToShowOnChart)
                #print(maxNumFaces)
                for x in range(maxNumFaces + 1):
                    if not (x in detectedObjectsDictionary):
                        obj = HistogramModal(str(x), 0, "", 0)
                        backend.histListModalFaceDetect_.appendToModel(obj)
                    else:
                        obj = HistogramModal(str(x), detectedObjectsDictionary[x].count, detectedObjectsDictionary[x].folderPath, detectedObjectsDictionary[x].count/maxValueToShowOnChart)
                        backend.histListModalFaceDetect_.appendToModel(obj)

            backend.runningThreads.pop()
            if not (bool(backend.runningThreads)):
                if(self.detectFacesAlgroBackend == 'azure') :
                    backend.scanWithoutIPUProcessingCompleted.emit()
                if(self.detectFacesAlgroBackend == 'ipu'):
                    backend.scanProcessingCompleted.emit()

        def run(self):
            #print("IN RUNNABLE RUN")
            self.run2(self.backend, self.inputFolderPath)
    #end performFaceDetection
    
    @Slot(QUrl)
    def startScan(self,urlpath):
        self.histListModal_.clearModel()
        self.histListModalFaceDetect_.clearModel()
        # Pass the current time to QML.
        print('startScan called', urlpath, urlpath.toLocalFile())
        inputFolderPath = urlpath.toLocalFile()

        countTotalFiles = 0
        for fileName in os.listdir(inputFolderPath):
            if os.path.isfile(os.path.join(inputFolderPath, fileName)):
                countTotalFiles = countTotalFiles + 1

        self.setTotalCount(countTotalFiles)

        self.shouldAbort = False
        self.runningThreads = list()
        runnable = self.performImageDetection(self, inputFolderPath, detetImagesAlgroBackend='ipu')
        runnableFaceDetect = self.performFaceDetection(self, inputFolderPath, detectFacesAlgroBackend='ipu')
        QThreadPool.globalInstance().start(runnable)
        QThreadPool.globalInstance().start(runnableFaceDetect)


    def performStopScan(self):
        self.shouldAbort = True
        self.histListModal_.clearModel()
        self.histListModalFaceDetect_.clearModel()
        self.setCurrentProcessedCount(0)
        self.setTotalCount(0)

    @Slot(QUrl)
    def stopScan(self, urlpath):
        self.performStopScan()
        print('stopScan called', urlpath, urlpath.toLocalFile())

    @Slot(QUrl)
    def startScanWithoutIPU(self, urlpath):
        # Pass the current time to QML.
        print('startScanWithoutIPU called', urlpath, urlpath.toLocalFile())
        
        self.histListModal_.clearModel()
        self.histListModalFaceDetect_.clearModel()
        # Pass the current time to QML.
        print('startScan called', urlpath, urlpath.toLocalFile())
        inputFolderPath = urlpath.toLocalFile()

        countTotalFiles = 0
        for fileName in os.listdir(inputFolderPath):
            if os.path.isfile(os.path.join(inputFolderPath, fileName)):
                countTotalFiles = countTotalFiles + 1

        self.setTotalCount(countTotalFiles)

        self.shouldAbort = False
        self.runningThreads = list()
        runnable = self.performImageDetection(self, inputFolderPath, detetImagesAlgroBackend='azure')
        runnableFaceDetect = self.performFaceDetection(self, inputFolderPath, detectFacesAlgroBackend='azure')
        QThreadPool.globalInstance().start(runnable)
        QThreadPool.globalInstance().start(runnableFaceDetect)
        #print("HELLO WORLD")


    @Slot(QUrl)
    def stopScanWithoutIPU(self, urlpath):
        self.performStopScan()
        print('stopScanWithoutIPU called', urlpath, urlpath.toLocalFile())

    @Slot(QUrl, result=str)
    def getDirectory(self, urlpath):
        return urlpath.toLocalFile()

    
    @Slot(str)
    def openFolder(self, urlpath):
        os.startfile(urlpath)

    @Slot(result=bool)
    def checkInternetConnection(self):
        print('checkInternetConnection called')
        url = "https://www.google.com/"
        timeout = 10
        try:
            request = requests.get(url,timeout=timeout)
            return True
        except Exception as exep:
            print(exep)
            return False

    @Slot(QUrl,result=int)
    def getImageCount(self, urlpath):
        print('getImageCount called')
        try:
            workDir = QDir(urlpath.toLocalFile())
            #print("urlpath.toLocalFile():"+urlpath.toLocalFile())
            filters = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]
            workDir.setFilter(QDir.Files | QDir.NoSymLinks | QDir.NoDotAndDotDot)
            workDir.setNameFilters(filters)
            #print("retCount:", len(workDir.entryList()))
            return len(workDir.entryList())
        except Exception as exep:
            print(exep)
            return 0

if __name__ == '__main__':
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    histogramListModal = HistogramListModal()
    histogramListModalFaceDetect = HistogramListModal()
    backend = Backend(histogramListModal, histogramListModalFaceDetect)
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("histListModal", histogramListModal)
    engine.rootContext().setContextProperty("histListModalFaceDetect", histogramListModalFaceDetect)

    QMLdirectory = os.path.dirname(__file__)
    engine.load(os.path.join(QMLdirectory, 'buildevent.qml'))
    app.setWindowIcon(QIcon(os.path.join(QMLdirectory,"images", "AMDGenericIcon.ico")))
    app.exec()
