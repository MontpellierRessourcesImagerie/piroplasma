import torch
import torchvision.ops as ops
import numpy as np
from qtpy.QtCore import QObject
from microglia_analyzer.tiles.tiler import ImageTiler2D



class PiroplasmaInference(QObject):


    def __init__(self, model, patchSize=1600, overlap=1/3.0):
        super(PiroplasmaInference, self).__init__()
        self.model = model
        self.patchSize = patchSize
        self.overlap = overlap
        self.imageSize = 1280
        self.distanceFromBorder = 10
        self.iouThreshold = 0.85
        self.tiler = None
        self.image = None
        self.tiles = None
        self.results = []
        self.boxes = torch.tensor([])
        self.scores = torch.tensor([])
        self.labels = torch.tensor([])
        self.minScore = 0


    def run(self):
        self.predict(self.image)


    def predict(self, image):
        self.createTiles(image)
        self.results = []
        for tile in self.tiles:
            self.results.append(self.model.predict(tile, imgsz=self.imageSize)[0])
        self.removeResultsCloseToTileBorders()
        self.removeResultsBelowMinScore()
        self.boxes, self.scores, self.labels = self.getFlattenedGlobalBoxScoreAndLabelTensors()
        self.supressNonMaximumBoxes()


    def removeResultsCloseToTileBorders(self):
        for result in self.results:
            result.boxes = [bbox for bbox in result.boxes if bbox.cls.int().item() != 3]
        for result in self.results:
            result.boxes = [bbox for bbox in result.boxes if
            self.distanceFromBorder < bbox.xyxy.tolist()[0][0] < self.patchSize - self.distanceFromBorder and
            self.distanceFromBorder < bbox.xyxy.tolist()[0][1] < self.patchSize - self.distanceFromBorder and
            self.distanceFromBorder < bbox.xyxy.tolist()[0][2] < self.patchSize - self.distanceFromBorder and
            self.distanceFromBorder < bbox.xyxy.tolist()[0][3] < self.patchSize - self.distanceFromBorder]


    def removeResultsBelowMinScore(self):
        for result in self.results:
            result.boxes = [bbox for bbox in result.boxes if bbox.conf.float().item() >= self.minScore]


    def getFlattenedGlobalBoxScoreAndLabelTensors(self):
        flattenedGlobalBoxes = []
        flattenedScores = []
        flattenedLabels = []
        for i, result in enumerate(self.results):
            y, x = self.tiler.layout[i].ul_corner
            if not result.boxes:
                continue
            boxes = [[int(round(box.xyxy.tolist()[0][0] + x)),
                          int(round(box.xyxy.tolist()[0][1] + y)),
                          int(round(box.xyxy.tolist()[0][2] + x)),
                          int(round(box.xyxy.tolist()[0][3] + y))
                          ] for box in result.boxes]
            scores = [box.conf.tolist()[0] for box in result.boxes]
            labels = [int(box.cls.tolist()[0]) for box in result.boxes]
            flattenedGlobalBoxes.extend(boxes)
            flattenedScores.extend(scores)
            flattenedLabels.extend(labels)
        return  (torch.tensor(flattenedGlobalBoxes, dtype=torch.float32),
                torch.tensor(flattenedScores, dtype=torch.float32),
                torch.tensor(flattenedLabels, dtype=torch.uint8))


    def supressNonMaximumBoxes(self):
        keepIndicesTensor = ops.nms(self.boxes, self.scores, iou_threshold=self.iouThreshold)
        self.boxes = self.boxes[keepIndicesTensor]
        self.labels = self.labels[keepIndicesTensor]
        self.scores = self.scores[keepIndicesTensor]


    def getAnnotationsPerLabel(self):
        annotationsByClass = AnnotationsByClass()
        for label, name in self.results[0].names.items():
            annotation = Annotations(label, name)
            annotationsByClass.addAnnotation(annotation)
        for box, score, label in zip(self.boxes.tolist(), self.scores.tolist(), self.labels.tolist()):
            annotation = annotationsByClass.get(label)
            if score < self.minScore:
                continue
            shape = [(int(round(box[1])),
                     int(round(box[0]))),
                     (int(round(box[3])),
                      int(round(box[2])))]
            annotation.addBox(shape)
            annotation.addScore(score)
        return annotationsByClass


    def batchPredict(self, folder):
        pass


    def createTiles(self, image):
        self.tiler = ImageTiler2D(self.patchSize, int(round(self.patchSize * self.overlap)), image.shape)
        self.tiles = self.tiler.image_to_tiles(image, False)


    def getYoloBBoxes(self):
        annotations = []
        imageHeight, imageWidth = self.tiler.shape[:2]
        print("image shape hxw", imageHeight, imageWidth)
        for score, box, label in zip(self.scores, self.boxes, self.labels):
            # Calculate the center, width, and height of the shape
            x_min, y_min = map(int, (box[0], box[1]))
            x_max, y_max = map(int, (box[2], box[3]))

            # Clip the coordinates to the image boundaries
            y_min = np.clip(y_min, 0, imageHeight - 1)
            y_max = np.clip(y_max, 0, imageHeight - 1)
            x_min = np.clip(x_min, 0, imageWidth - 1)
            x_max = np.clip(x_max, 0, imageWidth - 1)

            x_center = ((x_max + x_min) / 2) / imageWidth
            y_center = ((y_max + y_min) / 2) / imageHeight
            width = abs((x_max - x_min) / imageWidth)
            height = abs((y_max - y_min) / imageHeight)

            # Append the annotation to the list
            # class_name = shapes_layer.features["class"][i].split(":")[1].strip()
            # class_id = list(items_dict.keys())[list(items_dict.values()).index(class_name)]
            annotations.append(f"{label} {x_center} {y_center} {width} {height}")
        return annotations


    def getYoloBBoxesAsString(self):
        bboxes = self.getYoloBBoxes()
        bboxesString = "\n".join(bboxes)
        return bboxesString


    def saveYoloBBoxesTo(self, path):
        bboxes = self.getYoloBBoxesAsString()
        with open(path, 'w') as f:
            f.write(bboxes)


class Annotations(object):

    def __init__(self, label, name):
        super(Annotations, self).__init__()
        self.label = label
        self.name = name
        self.scores = []
        self.boxes = []

    def addBox(self, box):
        self.boxes.append(box)

    def addScore(self, score):
        self.scores.append(score)

    def getBoxes(self):
        return self.boxes

    def getLabel(self):
        return self.label

    def getName(self):
        return self.name

    def getScores(self):
        return self.scores



class AnnotationsByClass(object):

    def __init__(self):
        super(AnnotationsByClass, self).__init__()
        self.annotationsPerClass = {}
        self.index = 0

    def addAnnotation(self, annotation):
        self.annotationsPerClass[annotation.label] = annotation

    def getListOfLabels(self):
        return list(self.annotationsPerClass.keys())

    def get(self, label):
        return self.annotationsPerClass[label]

    def getSize(self):
        return len(self.annotationsPerClass.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.annotationsPerClass.keys()):
            result = self.annotationsPerClass[list(self.annotationsPerClass.keys())[self.index]]
            self.index += 1
            return result
        else:
            raise StopIteration
