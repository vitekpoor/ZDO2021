import pytest
import os
import skimage.io
import glob
import numpy as np
from pathlib import Path
import zdo2021.main

# cd ZDO2021
# python -m pytest

def test_run_all():
    """
    Run varroa detection algorithm for all images from test dataset.
    """

    vdd = zdo2021.main.VarroaDetector()

    # Nastavte si v operačním systém proměnnou prostředí 'VARROA_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = os.getenv('VARROA_DATA_PATH_', default=Path(__file__).parent / 'test_dataset/')

    # print(f'dataset_path = {dataset_path}')
    files = glob.glob(f'{dataset_path}/images/*.jpg')

    # create array of all images
    images = []

    # read all images
    for filename in files:
        # read image 
        im = skimage.io.imread(filename)
        # collect all loaded images
        images.append(im)
    
    # create data for prediction
    imgs = np.stack(images, axis=0)

    # make prediction
    #prediction = vdd.predict(imgs)

    # make assertion
    #for i in range(len(prediction)): 
    #    assert prediction[i].shape[0] == imgs[i].shape[0]

    # Toto se bude spouštět všude mimo GitHub
    if not os.getenv('CI'):
        import matplotlib.pyplot as plt
        import json
        
        with open(dataset_path.__str__() + "\\annotations\\instances_default.json") as json_file:
            gt_ann = json.load(json_file)
        
        for filename in files:

            name = os.path.basename(filename)

            image_id = -1

            for i in gt_ann["images"]:
                if i["file_name"] == name:
                    image_id = i["id"]
                    break

            if image_id == -1:
                continue

            #ravel = prediction[0].ravel()
            #a, b, c = plt.hist(ravel, 40, density=False)

            im = skimage.io.imread(filename)

            plt.figure(filename)
            plt.imshow(im, cmap='gray')
            #plt.show()

            for i in gt_ann["annotations"]:
                if i["image_id"] == image_id:
                    segmentation = i["segmentation"][0]
                    xs = []
                    ys = []
                    index = 0
                    for j in segmentation:
                        if(index % 2 == 0):
                            ys.append(j)
                        else:
                            xs.append(j)
                        index = index + 1
                    plt.plot(xs, ys, color="red", linewidth=0.5) 

            output = "annot\\" + name

            plt.savefig(output, dpi = 600)


def test_run_random():
    vdd = zdo2021.main.VarroaDetector()

    # Nastavte si v operačním systém proměnnou prostředí 'VARROA_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = os.getenv('VARROA_DATA_PATH_', default=Path(__file__).parent / 'test_dataset/')

    # print(f'dataset_path = {dataset_path}')
    files = glob.glob(f'{dataset_path}/images/*.jpg')
    cislo_obrazku = np.random.randint(0, len(files))
    filename = files[cislo_obrazku]

    im = skimage.io.imread(filename)
    imgs = np.expand_dims(im, axis=0)
    # print(f"imgs.shape={imgs.shape}")
    prediction = vdd.predict(imgs)


    assert prediction.shape[0] == imgs.shape[0]


    # Toto se bude spouštět všude mimo GitHub
    if not os.getenv('CI'):
        import matplotlib.pyplot as plt
        import json
        
        with open(dataset_path.__str__() + "\\annotations\\instances_default.json") as json_file:
            gt_ann = json.load(json_file)
        
        name = os.path.basename(filename)

        for i in gt_ann["images"]:
            if i["file_name"] == name:
                image_id = i["id"]
                break

        #ravel = prediction[0].ravel()
        #a, b, c = plt.hist(ravel, 40, density=False)

        plt.figure(filename)
        plt.imshow(prediction[0], cmap='gray')
        #plt.show()

        for i in gt_ann["annotations"]:
            if i["image_id"] == image_id:
                segmentation = i["segmentation"][0]
                xs = []
                ys = []
                index = 0
                for j in segmentation:
                    if(index % 2 == 0):
                        ys.append(j)
                    else:
                        xs.append(j)
                    index = index + 1
                plt.plot(xs, ys, color="yellow", linewidth=0.1) 

        output = "results\\" + name

        plt.savefig(output, dpi = 600)

        os.startfile(output)

        #assert f1score(ground_true_masks, prediction) > 0.55
        
        

def f1score(gt_ann, prediction):
    pass

def prepare_ground_true_masks(gt_ann, filname):
    pass