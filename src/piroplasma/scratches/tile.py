from tifffile import imread
from piroplasma.inference import PiroplasmaInference
from ultralytics import YOLO
from PIL import Image


model = YOLO('/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/runs/detect/train-10/weights/best.pt')
PATH = "/media/baecker/data/2026/in/bbs/real_data/K002946_130524_100Xmosaic_5-5_bis.tif"
image = imread(PATH)
print("is rgb", image.shape[-1]==3)
inference = PiroplasmaInference(model, patchSize=1600)
inference.predict(image)
print(inference.patches)
img = Image.fromarray(inference.patches[0], 'RGB')
img.show()

