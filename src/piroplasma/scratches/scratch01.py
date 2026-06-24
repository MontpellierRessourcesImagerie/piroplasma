from tifffile import imread

PATH = "/media/baecker/data/2026/in/bbs/real_data/K002946_130524_100Xmosaic_5-5_bis.tif"
image = imread(PATH)
ddImage = image[:,:,0]

w, h, = image.shape[:2]
w2, h2 = ddImage.shape[:2]

print(w, h, w2, h2)