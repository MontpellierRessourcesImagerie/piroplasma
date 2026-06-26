from microglia_analyzer.tiles.tiler import ImageTiler2D
from ultralytics import YOLO

class PiroplasmaInference(object):


    def __init__(self, model, patchSize=1600, overlap=None):
        super(PiroplasmaInference, self).__init__()
        self.model = model
        self.patchSize = patchSize
        self.imageSize = 1280
        self.distanceFromBorder = 100
        self.tiler = None
        if overlap is None:
            self.overlap = int(round(self.patchSize / 3.0))
        self.tiles = None
        self.yoloOutput = []
        self.results = []


    def predict(self, image):
        self.createTiles(image)
        self.results = []
        for tile in self.tiles:
            self.results.append(self.model.predict(tile, imgsz=self.imageSize)[0])
        for result in self.results:
            result.boxes = [bbox for bbox in result.boxes if
                    self.distanceFromBorder < bbox.xyxy.tolist()[0][0] < self.patchSize - self.distanceFromBorder and
                    self.distanceFromBorder < bbox.xyxy.tolist()[0][1] < self.patchSize - self.distanceFromBorder and
                    self.distanceFromBorder < bbox.xyxy.tolist()[0][2] < self.patchSize - self.distanceFromBorder and
                    self.distanceFromBorder < bbox.xyxy.tolist()[0][3] < self.patchSize - self.distanceFromBorder]
        self.yoloOutput = []
        for i, result in enumerate(self.results):
            y, x = self.tiler.layout[i].ul_corner
            result.boxes = [[box[1] + y, box[0] + x, box[3] + y, box[2] + x] for box in result.boxes]
            scores = [box.conf.tolist()[0] for box in result.boxes]
            classes = [int(box.cls.tolist()[0]) for box in result.boxes]
            for aBox, aScore, aClass in zip(result.boxes, scores, classes):
                self.yoloOutput.append((aBox, aScore, aClass))
        '''            
        for i, tileResults in enumerate(results.xyxy):
            boxes = tileResults[:, :4].tolist()
            boxes = [[f * a, f * b, f * c, f * d] for (a, b, c, d) in boxes]
            y, x = tiles_manager.layout[i].ul_corner
            y, x = int(y * f), int(x * f)
            boxes = [[box[1] + y, box[0] + x, box[3] + y, box[2] + x] for box in boxes]
            scores = tileResults[:, 4].tolist()
            classes = [int(c) for c in tileResults[:, 5].tolist()]
            for box, score, c in zip(boxes, scores, classes):
                self.yolo_output.append((box, score, c))
        '''

    def batchPredict(self, folder):
        pass


    def createTiles(self, image):
        self.tiler = ImageTiler2D(self.patchSize, self.overlap, image.shape)
        self.tiles = self.tiler.image_to_tiles(image, False)