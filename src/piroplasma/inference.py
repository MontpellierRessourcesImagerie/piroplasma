from microglia_analyzer.tiles.tiler import ImageTiler2D


class PiroplasmaInference(object):

    def __init__(self, model, patchSize=1600, overlap=None):
        super(PiroplasmaInference, self).__init__()
        self.model = model
        self.patchSize = patchSize
        if overlap is None:
            self.overlap = int(round(self.patchSize / 3.3333333))
        self.patches = None


    def predict(self, image):
        self.createTiles(image)


    def batchPredict(self, folder):
        pass


    def createTiles(self, image):
        tiler = ImageTiler2D(self.patchSize, self.overlap, image.shape)
        self.patches = tiler.image_to_tiles(image, False)