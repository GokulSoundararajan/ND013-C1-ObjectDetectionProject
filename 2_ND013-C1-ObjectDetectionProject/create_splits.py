import argparse
import glob
import os


import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    trainDir = data_dir + '/train'
    validationDir =  data_dir + '/val'
    testDir = data_dir + '/test'

    try:
        # Create target Directories
        os.mkdir(trainDir)
        print(trainDir + " Created")
        os.mkdir(validationDir)
        print(validationDir + " Created")
        os.mkdir(testDir)
        print(testDir + " Created") 
    except FileExistsError:
        print("Directory already exists")

    fileArray = glob.glob(data_dir)


    
    #Split the .tfrecord files in different folders in the ratio 80:10:10. 
    for name in fileArray:
        if( (fileArray.index(name)+1)/len(fileArray)*100 <= 80 and os.path.isfile(name) ):
            head_tail = os.path.split(name)
            newpath = os.path.join(trainDir, head_tail[1])
            os.rename(name, newpath)
        elif( (fileArray.index(name)+1)/len(fileArray)*100 > 80 and (fileArray.index(name)+1)/len(fileArray)*100 <= 90 and os.path.isfile(name)):
            head_tail = os.path.split(name)
            newpath = os.path.join(validationDir,  head_tail[1])
            os.rename(name, newpath)
        else:
            if(os.path.isfile(name)):
                head_tail = os.path.split(name)
                newpath = os.path.join(testDir, head_tail[1])
                os.rename(name, newpath)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)