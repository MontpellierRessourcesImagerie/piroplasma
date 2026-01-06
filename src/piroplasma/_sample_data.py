"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import os
import napari
import numpy as np
import appdirs
import requests
from qtpy.QtCore import QObject
from napari.qt.threading import create_worker
from skimage import io


class Downloader(QObject):


    def __init__(self, url, path):

        super().__init__(None)
        self.chunkSize = 1024*1024
        self.url = url
        self.path = path


    def run(self):
        session = requests.Session()
        r = session.get(self.url, stream=True)
        r.raise_for_status()
        tmpPath = self.path.replace(".tif", "_tmp")
        if os.path.exists(tmpPath):
            os.remove(tmpPath)
        with open(tmpPath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=self.chunkSize):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    yield



def getCacheDir():
    cache_dir = os.path.join(appdirs.user_cache_dir(), "piroplasma")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def getImagePath():
    image_path = os.path.join(getCacheDir(), "example_image_001.tif")
    return image_path


def downloadFinished():
    imagePath = getImagePath()
    tmpPath = imagePath.replace('.tif', '_tmp')
    os.rename(tmpPath, imagePath)
    napari.current_viewer().layers.remove('Downloading image...')
    return loadImage()


def loadImage():
    scale = (1, 1)
    image = io.imread(getImagePath())
    data = np.array(image)
    return [(data,
             {'scale': scale,
              'name': 'example (piroplasma example image)'}),
            ]


def make_sample_data():
    """Download and return a sample image for the piroplasma plugin.
    """
    url = 'https://zenodo.org/records/18154748/files/example_image_001.tif?download=1'
    image_path = getImagePath()
    if not os.path.exists(image_path):
        downloader = Downloader(url, image_path)
        chunks = 1287565150 // downloader.chunkSize
        worker = create_worker(downloader.run,
                               _progress={'total': chunks, 'desc': "Downloading example image..."}
                               )
        worker.finished.connect(downloadFinished)
        worker.start()
        placeholder_image = np.zeros((1,1), np.uint8)
        return [(placeholder_image,
             {
              'name': 'Downloading image...'}),
            ]
    return loadImage()
