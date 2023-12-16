import os

def rename(path):
    FileList = os.listdir(path)
    for files in FileList:
        oldDirPath = os.path.join(path, files)
        fileName = os.path.splitext(files)[0]
        fileType = os.path.splitext(files)[1]
        newDirPath = os.path.join(path, "pos." + fileName + fileType)
        os.rename(oldDirPath, newDirPath)


path = "../concerete_crack_images/training/Positive/"
rename(path)