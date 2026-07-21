import napari
from qtpy.QtWidgets import QWidget
from piroplasma.inference import PiroplasmaInference



class RedBloodCellClassifierWidget(QWidget):

    def __init__(self, parent=None):
        super(RedBloodCellClassifierWidget, self).__init__(parent)
        self.viewer = napari.current_viewer()
        self.inference = PiroplasmaInference()