import numpy as np
import skimage
import skimage.measure
import json
import math
from scipy import ndimage
from skimage.filters import threshold_otsu
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
from . import podpurne_funkce

class VarroaDetector():

    def __init__(self):
        # load config file
        with open('config.json') as json_file:
            self.config = json.load(json_file)
            
    def predict(self, data):
        """
        Varroa detector.
        
        :param data: np.ndarray with shape [pocet_obrazku, vyska, sirka, barevne_kanaly]
        
        :return: shape [pocet_obrazku, vyska, sirka], 0 - nic, 1 - varroa destructor
        
        """
        
        # get information from data
        data_count = data.shape[0]
        data_height = data.shape[1]
        data_width = data.shape[2]
        data_channels = data.shape[3]
        
        # print information
        print("{}: {}".format("Images count", data_count))
        print("{}: {}x{}".format("Dimension", data_width, data_height))
        print("{}: {}".format("Color channels", data_channels))
        
        # create output based on input data
        output = []
        
        # proceed varroa detection in each image
        for i in range(data_count):
         
            # convert to grayscale
            img_gray = skimage.color.rgb2gray(data[i])
        
            img_blur = ndimage.gaussian_filter(img_gray, sigma=1)

            # threshold based on config value
            thresh = threshold_otsu(img_blur)
            #thresh = 0.25
            img_threshold = img_blur < thresh
        
            # get label image
            img_label = skimage.measure.label(img_threshold)
            
            # get region properties
            img_region_props = skimage.measure.regionprops(img_label)
        
            labels_to_remove = []

            for prop in img_region_props:
                
                circularity = (4 * math.pi * prop.area) / (prop.perimeter**2)
                
                if (
                    #prop.euler_number != 1 
                    circularity < 0.7 
                    or circularity > 1.1 
                    or prop.area < 100  # TODO !!! 
                    #or prop.area > 1000
                ):
                    labels_to_remove.append(prop.label)

            img_label[np.isin(img_label, labels_to_remove)] = 0

            final_mask =  img_threshold < img_label

            # write image to output
            output.append(final_mask)
        
        return np.array(output)