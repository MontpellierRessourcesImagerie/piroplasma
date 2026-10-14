import os

dataset = "ds260901"
baseFolder = "/run/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/"
datasetFolder = baseFolder + dataset + "/"
backgroundFolder = datasetFolder + "/background/"

IN_PATHS = [datasetFolder]
OUT_PATHS = [backgroundFolder]

PATHS = zip(IN_PATHS, OUT_PATHS)

def main():
    for inPath,outPath in PATHS:
        files = os.listdir(inPath)
        for file in files:
            if file.endswith(".tif"):
                txtFile = file.replace(".tif",".txt")
                if not txtFile in files:
                    tifPath = os.path.join(inPath,file)
                    tifOutPath = os.path.join(outPath,file)
                    os.rename(tifPath,tifOutPath)


if __name__ == "__main__":
    main()