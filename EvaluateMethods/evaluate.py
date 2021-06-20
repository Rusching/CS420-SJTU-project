import imagej
from skimage.io import imread_collection, imsave
import numpy as np
import os
import re

ij = imagej.init('Fiji.app')

Language_extension = "BeanShell"


macroVRand = """
import trainableSegmentation.metrics.*;
#@output String VRand
import ij.IJ;
originalLabels=IJ.openImage("AAAAA");
proposedLabels=IJ.openImage("BBBBB");
metric = new RandError( originalLabels, proposedLabels );
maxThres = 1.0;
maxScore = metric.getMaximalVRandAfterThinning( 0.0, maxThres, 0.1, true );  
VRand = maxScore;
"""

macroVInfo = """
import trainableSegmentation.metrics.*;
#@output String VInfo
import ij.IJ;
originalLabels=IJ.openImage("AAAAA");
proposedLabels=IJ.openImage("BBBBB");
metric = new VariationOfInformation( originalLabels, proposedLabels );
maxThres =1.0;
maxScore = metric.getMaximalVInfoAfterThinning( 0.0, maxThres, 0.1 );  
VInfo = maxScore;
"""

def evl(image_dir, label_dir, i):
    # global macroVRand
    # global macroVInfo
    image_list = os.listdir(image_dir)
    label_list = os.listdir(label_dir)
    image_path = os.path.join(image_dir, image_list[i]).replace('\\', '/')
    label_path = os.path.join(label_dir, label_list[i]).replace('\\', '/')
    reg1 = re.compile('AAAAA')
    macror = reg1.sub(label_path, macroVRand)
    macroi = reg1.sub(label_path, macroVInfo)

    reg2 = re.compile('BBBBB')
    macror = reg2.sub(image_path, macror)
    macroi = reg2.sub(image_path, macroi)

    VRand = float(str(ij.py.run_script(Language_extension, macror).getOutput('VRand')))
    VInfo = float(str(ij.py.run_script(Language_extension, macroi).getOutput('VInfo')))
    print(os.path.basename(image_path) + ' max rand error {}, max info error{}'.format(str(VRand), str(VInfo)))
    with open('result_1.txt', 'a') as f:
        f.write(os.path.basename(image_path) + ' max rand error {}, max info error{}'.format(str(VRand), str(VInfo)) + '\n')
    return VRand, VInfo
