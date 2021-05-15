import numpy as np
import skimage
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
from . import podpurne_funkce

class VarroaDetector():

    def __init__(self):
        pass

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
       output = np.zeros(shape=(data_count, data_height, data_width))

       # proceed varroa detection in each image
       for i in range(data_count):

           # convert to grayscale
           imggray = skimage.color.rgb2gray(data[i])

           # write image to output
           output[i] = imggray

       return output