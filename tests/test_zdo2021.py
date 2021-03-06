import pytest
import os
import skimage.io
import glob
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import json
from pathlib import Path
from skimage.draw import polygon

import zdo2021.main

# cd ZDO2021
# python -m pytest

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
        
        ann_pth = Path(dataset_path)/"annotations/instances_default.json"
        assert ann_pth.exists()
        # gt_ann = json.loads(str(ann_pth))
        with open(ann_pth, 'r') as infile:
            gt_ann = json.load(infile)

        ground_true_mask = prepare_ground_true_mask(gt_ann, filename)
        
        plt.imshow(prediction[0], cmap='gray')
        plt.contour(ground_true_mask)
        plt.show()

        f1 = f1score(ground_true_mask, prediction)

        print(f"f1score={f1}")
        # assert f1 > 0.55
        
        

def test_run_all():
    vdd = zdo2021.main.VarroaDetector()

    # Nastavte si v operačním systém proměnnou prostředí 'VARROA_DATA_PATH' s cestou k datasetu.
    # Pokud není nastavena, využívá se testovací dataset tests/test_dataset
    dataset_path = os.getenv('VARROA_DATA_PATH_', default=Path(__file__).parent / 'test_dataset/')
    # dataset_path = Path(r"H:\biology\orig\zdo_varroa_detection_coco_001")

    # print(f'dataset_path = {dataset_path}')
    files = glob.glob(f'{dataset_path}/images/*.jpg')
    f1s = []
    for filename in files:
        im = skimage.io.imread(filename)
        imgs = np.expand_dims(im, axis=0)
        # print(f"imgs.shape={imgs.shape}")
        prediction = vdd.predict(imgs)

        ann_pth = Path(dataset_path)/"annotations/instances_default.json"
        assert ann_pth.exists()
        # gt_ann = json.loads(str(ann_pth))
        with open(ann_pth, 'r') as infile:
            gt_ann = json.load(infile)

        ground_true_mask = prepare_ground_true_mask(gt_ann, filename)
        
        # plt.imshow(prediction[0])
        # plt.contour(ground_true_mask)
        # plt.show()

        f1i = f1score(ground_true_mask, prediction)
        # assert f1i > 0.55
        f1s.append(f1i)

    f1 = np.mean(f1s)
    print(f"f1score={f1}")
    print(f1s)
    # assert f1 > 0.55


def f1score(ground_true_mask:np.ndarray, prediction:np.ndarray):
    """
    Measure f1 score for one image
    :param ground_true_mask:
    :param prediction:
    :return:
    """
    f1 = sklearn.metrics.f1_score(ground_true_mask.astype(bool).flatten(), prediction.astype(bool).flatten())
    return f1


def prepare_ground_true_mask(gt_ann, filename):
    name = None
    for ann_im in gt_ann['images']:
        if  ann_im["file_name"] == Path(filename).name:
            # mask = np.zeros([], dtype=bool)
            M = np.zeros((ann_im["width"], ann_im["height"]), dtype=bool)
            immage_id = ann_im["id"]
            for ann in gt_ann['annotations']:
                if ann["image_id"] == immage_id:
                    S = ann['segmentation']
                    for s in S:
                        N = len(s)
                        rr, cc = polygon(np.array(s[0:N:2]), np.array(s[1:N:2]))  # (y, x)
                        M[rr, cc] = True
    return M