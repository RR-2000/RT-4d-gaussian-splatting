from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
import cv2 as cv
import glob
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        resolution_scale = 1
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.resolution_scale = resolution_scale

    def __getitem__(self, index):

        caminfo = self.dataset[index]
        cam = loadCam(self.args, index, caminfo, self.resolution_scale)
        return cam.image, cam

    def __len__(self):
        
        return len(self.dataset)


class evalDataset(Dataset):
    def __init__(
        self, model_path
    ):
        self.model_path = model_path
        self.gt_paths = glob.glob(model_path+'/gt/*.png')
        
    def __getitem__(self, index):

        gt_path = self.gt_paths[index]

        return self.load_image_pair(gt_path)

    def __len__(self):
        
        return len(self.gt_paths)

    def load_image(self, path):
        try:
            image = cv.cvtColor(cv.imread(path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)
        except:
            raise FileNotFoundError(f"{path} does not exist")

        return image

    def load_image_pair(self, path):
    
        gt_path = path
        pred_path = path.replace("gt", "renders")

        gt =  self.load_image(gt_path)
        pred = self.load_image(pred_path)

        return gt, pred
