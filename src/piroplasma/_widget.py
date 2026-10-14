import os.path

import appdirs
import napari
from napari.utils import notifications
from ultralytics import YOLO
from qtpy.QtWidgets import QWidget
from qtpy.QtGui import QColor
from napari.qt.threading import create_worker
from autooptions import (Options,
                         OptionsWidget)
from piroplasma.inference import PiroplasmaInference



class RedBloodCellClassifierWidget(QWidget):


    def __init__(self, parent=None):
        super(RedBloodCellClassifierWidget, self).__init__(parent)
        self.viewer = napari.current_viewer()
        self.options = self.getOptions()
        self.optionsWidget = None
        self.operation = None
        self.imageLayer = None
        self.display()


    @classmethod
    def getOptions(cls):
        options = Options("Piroplasma", "Red Blood Cell Classifier")
        options.addImage()
        options.addInt("Patch Size", value=1600)
        options.addInt("Image Size", value=1280)
        options.addFloat("Tile Overlap", value=0.3333)
        options.addFloat("IoU Threshold", 0.7)
        options.addFloat("Min. Score", 0.0)
        options.addFile(name="Model", transient=False)
        options.load()
        return options


    def display(self):
        self.optionsWidget = OptionsWidget(self.viewer, self.options, layout_type="vertical", client=self)
        self.optionsWidget.addApplyButton(self.applyButtonPressed)
        self.viewer.window.add_dock_widget(self.optionsWidget, name=self.options.optionsName)


    def applyButtonPressed(self):
        self.apply()


    def apply(self):
        self.imageLayer = self.optionsWidget.getImageLayer("image")
        if not self.imageLayer:
            notifications.show_error("No image selected!")
            return
        if not os.path.exists(self.options.value("Model")):
            notifications.show_error("Model not found!")
            return
        model = YOLO(self.options.value("Model"))
        self.operation = PiroplasmaInference(model,
                                             self.options.value("Patch Size"),
                                             self.options.value("Tile Overlap")
                                             )
        self.operation.imageSize = self.options.value("Image Size")
        self.operation.iouThreshold = self.options.value("IoU Threshold")
        self.operation.minScore = self.options.value("Min. Score")
        self.operation.image = self.imageLayer.data
        self.runOperationInThread("Classifying Red Blood Cells...", self.onClassificationFinished)


    def runOperationInThread(self, description, callback=None):
        worker = create_worker(
            self.operation.run, _progress={"desc": description}
        )
        if callback is not None:
            worker.finished.connect(callback)
        worker.start()


    def onClassificationFinished(self):
        annotations = self.operation.getAnnotationsPerLabel()
        colors = ["red", "green", "blue", "cyan", "magenta", "yello", "black", "white"]
        for annotation in annotations:
            boxes = annotation.getBoxes()
            label = annotation.getLabel()
            name = annotation.getName()
            scores = annotation.getScores()
            self.viewer.add_shapes(boxes,
                                   name=name,
                                   edge_color=colors[label],
                                   face_color=QColor(1,1,1,0).name() + "00",
                                   edge_width=8,
                                   properties=scores)
        path = os.path.splitext(self.imageLayer.source.path)[0] + ".txt"
        self.operation.saveYoloBBoxesTo(path)
