import os

trainPath = "/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/work/train/"
trainBackgroundPath = "/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/work/background/train/"
valPath = "/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/work/val/"
valBackgroundPath = "/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/work/background/val/"
testPath = "/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/work/test/"
testBackgroundPath = "/media/baecker/data/programs/Fiji/mri-tools/piroplasma/src/datasets/work/background/test/"

IN_PATHS = [trainPath, valPath, testPath]
OUT_PATHS = [trainBackgroundPath, valBackgroundPath, testBackgroundPath]
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