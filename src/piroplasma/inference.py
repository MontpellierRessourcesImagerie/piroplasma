import torch
import torchvision.ops as ops
from microglia_analyzer.tiles.tiler import ImageTiler2D



class PiroplasmaInference(object):


    def __init__(self, model, patchSize=1600, overlap=None):
        super(PiroplasmaInference, self).__init__()
        self.model = model
        self.patchSize = patchSize
        self.imageSize = 1280
        self.distanceFromBorder = 100
        self.iouThreshold = 0.85
        self.tiler = None
        if overlap is None:
            self.overlap = int(round(self.patchSize / 3.0))
        self.tiles = None
        self.results = []
        self.boxes = torch.tensor([])
        self.scores = torch.tensor([])
        self.labels = torch.tensor([])
        self.minScore = 0


    def predict(self, image):
        self.createTiles(image)
        self.results = []
        for tile in self.tiles:
            self.results.append(self.model.predict(tile, imgsz=self.imageSize)[0])
        self.removeResultsCloseToTileBorders()
        self.boxes, self.scores, self.labels = self.getFlattenedGlobalBoxScoreAndLabelTensors()
        self.supressNonMaximumBoxes()


    def removeResultsCloseToTileBorders(self):
        for result in self.results:
            result.boxes = [bbox for bbox in result.boxes if
            self.distanceFromBorder < bbox.xyxy.tolist()[0][0] < self.patchSize - self.distanceFromBorder and
            self.distanceFromBorder < bbox.xyxy.tolist()[0][1] < self.patchSize - self.distanceFromBorder and
            self.distanceFromBorder < bbox.xyxy.tolist()[0][2] < self.patchSize - self.distanceFromBorder and
            self.distanceFromBorder < bbox.xyxy.tolist()[0][3] < self.patchSize - self.distanceFromBorder]


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


    def getBoxesAsShapesPerLabel(self):
        healthyShapes = []
        infectedShapes = []
        poppedShapes = []
        for box, score, label in zip(self.boxes.tolist(), self.scores.tolist(), self.labels.tolist()):
            if score < self.minScore:
                continue
            shape = [(int(round(box[1])),
                     int(round(box[0]))),
                     (int(round(box[3])),
                      int(round(box[2])))]
            if label == 0:
                healthyShapes.append(shape)
            if label == 1:
                infectedShapes.append(shape)
            if label == 2:
                poppedShapes.append(shape)
        return healthyShapes, infectedShapes, poppedShapes


    def batchPredict(self, folder):
        pass


    def createTiles(self, image):
        self.tiler = ImageTiler2D(self.patchSize, self.overlap, image.shape)
        self.tiles = self.tiler.image_to_tiles(image, False)