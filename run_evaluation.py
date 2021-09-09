import pickle
from typing import List

import os
import shutil
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from mindspore.common.parameter import ParameterTuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from easydict import EasyDict as ed
import participant_model
import code
import mindspore as ms
def create_model(model_checkpoint_file):
    # we expect the participant_model is in same directory and is called a participant_model.py
    model = participant_model.Net()
    param_dict = ms.load_checkpoint(str(model_checkpoint_file))
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    return model



def read_image(
    image_dir: Path
) -> List[ms.Tensor]:
    images = []
    image_ids = []
    for image_path in image_dir.iterdir():
        image = cv2.imread(str(image_path)).transpose((2,0,1)) # (H, W, C) -> (C, H, W)
        images.append(image)
        image_ids.append(image_path.stem)
    return images, image_ids


def main():
    image_dir = Path(input())
    images, image_ids = read_image(image_dir)
    model = create_model(Path(input()))
    for iid, image in zip(image_ids, images):
        image = participant_model.pre_process(iid, image)
        prediction = model(**image)
        result = participant_model.post_process(iid, prediction)
        print(prediction)
        print(result)
        code.interact(local=locals())

if __name__ == '__main__':
    main()