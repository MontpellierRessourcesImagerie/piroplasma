import os.path

import napari
from autooptions import Options
from autooptions import OptionsWidget
from napari.qt.threading import create_worker
from napari.utils import notifications
from qtpy.QtGui import QColor
import shutil


from qtpy.QtWidgets import QWidget, QVBoxLayout
from ultralytics import YOLO

from piroplasma.inference import PiroplasmaInference
from piroplasma.tiling import tile_dataset



class RedBloodCellClassifierWidget(QWidget):


    def __init__(self, parent=None):
        super(RedBloodCellClassifierWidget, self).__init__(parent)
        self.viewer = napari.current_viewer()
        self.options = self.getOptions()
        self.optionsWidget = None
        self.operation = None
        self.imageLayer = None
        self.createLayout()


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


    def createLayout(self):
        self.optionsWidget = OptionsWidget(self.viewer, self.options, layout_type="vertical", client=self)
        self.optionsWidget.addApplyButton(self.applyButtonPressed)
        layout = QVBoxLayout()
        layout.addWidget(self.optionsWidget)
        self.setLayout(layout)


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
                                   face_color=QColor(1, 1, 1, 0).name() + "00",
                                   edge_width=8,
                                   properties=scores)
        path = os.path.splitext(self.imageLayer.source.path)[0] + ".txt"
        self.operation.saveYoloBBoxesTo(path)



class ExportTrainingDataWidget(QWidget):
    """Widget to export training data by tiling images.
    Uses the autooptions framework for UI and runs ``tile_dataset`` in a
    background thread with a napari progress bar.
    """


    def __init__(self, parent=None):
        super(ExportTrainingDataWidget, self).__init__(parent)
        self.viewer = napari.current_viewer()
        self.options = self.getOptions()
        self.optionsWidget = None
        self.operation = None  # will hold a callable that runs tile_dataset
        self.result = None
        self.createLayout()


    @classmethod
    def getOptions(cls):
        opts = Options("Piroplasma", "Export Training Data")
        opts.addFolder(name="Input Folder", transient=False)
        opts.addFolder(name="Output Folder", transient=False)
        opts.addInt("Tile Size", value=1600)
        opts.addFloat("Tile Overlap", value=0.33)
        opts.addInt("Border Class", value=3)
        opts.addInt("Border Margin", value=0)
        opts.addInt("Min Box Size", value=200)
        opts.addInt("Global Margin", value=0)
        # image extensions – list of strings; use a free‑form text entry
        opts.addStr(
            "Image Extensions",
            value=".tif,.tiff,.png,.jpg,.jpeg"
        )
        opts.addBool("Overwrite", value=False)
        opts.load()
        return opts


    def createLayout(self):
        self.optionsWidget = OptionsWidget(
            self.viewer, self.options, layout_type="vertical", client=self
        )
        self.optionsWidget.addApplyButton(self.applyButtonPressed)
        layout = QVBoxLayout()
        layout.addWidget(self.optionsWidget)
        self.setLayout(layout)


    def applyButtonPressed(self):
        self.apply()


    def apply(self):
        # Validate input folder
        input_dir = self.options.value("Input Folder")
        output_dir = self.options.value("Output Folder")
        if not os.path.isdir(input_dir):
            notifications.show_error("Input folder does not exist!")
            return
        # Parse image extensions string into a list
        ext_str = self.options.value("Image Extensions")
        extensions = [e.strip() for e in ext_str.split(",") if e.strip()]
        # Build options dict for tile_dataset
        tile_opts = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "tile_size": self.options.value("Tile Size"),
            "overlap": self.options.value("Tile Overlap"),
            "border_class": self.options.value("Border Class"),
            "border_margin": self.options.value("Border Margin"),
            "image_extensions": extensions,
            "overwrite": self.options.value("Overwrite"),
            "min_box_size": self.options.value("Min Box Size"),
            "global_margin": self.options.value("Global Margin"),
        }

        global_margin = tile_opts["global_margin"]
        if global_margin < 0:
            notifications.show_error("Global Margin must not be negative!")
            return


        # Define the operation callable
        def _run():
            # Run tiling
            result = tile_dataset(**tile_opts)
            # Copy supplementary YAML files if present
            for fname in ("class.yaml", "colors.yaml"):
                src_path = os.path.join(input_dir, fname)
                if os.path.isfile(src_path):
                    try:
                        shutil.copy2(src_path, output_dir)
                    except Exception as e:
                        notifications.show_error(f"Failed to copy {fname}: {e}")
            return result


        self.operation = _run
        # Run in napari worker thread
        self.runOperationInThread(
            "Exporting training data…", self.onExportFinished
        )


    def runOperationInThread(self, description, callback=None):
        # reuse existing helper – it expects a callable with .run, adapt by wrapping
        worker = create_worker(
            self.operation, _progress={"desc": description}
        )
        if callback is not None:
            worker.finished.connect(callback)
        worker.start()


    def onExportFinished(self):
        # The result of tile_dataset is stored in worker.return (accessible via self.operation?)
        # create_worker stores the return value on the worker object; we can fetch it via the signal.
        # For simplicity, we re‑run tile_dataset with the same options via the saved callable and
        # capture the return value here (the worker already finished, so the callable has been executed).
        # We retrieve it from the worker's ``result`` attribute via a temporary variable.
        # Since we cannot access the worker here directly, we will re‑execute the callable to get the dict.
        try:
            result = self.operation()
        except Exception as e:
            notifications.show_error(f"Export failed: {e}")
            return
        count = len(result)
        notifications.show_info(f"Export finished – {count} tiles written.")
