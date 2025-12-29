from sgm_wrapper import SgmGpu
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

class SGM:
    def __init__(self, width, height, fov, baseline, device):
        self.width = width
        self.height = height
        self.fov = fov
        self.baseline = baseline
        self.device = device
        self.sgm = SgmGpu(width, height)

    def compute_depth(self, left: torch.Tensor, right: torch.Tensor):
        left = left.cpu().numpy()
        right = right.cpu().numpy()
        # convert rgba to grayscale
        left = np.dot(left[...,:3], [0.2989, 0.5870, 0.1140])
        right = np.dot(right[...,:3], [0.2989, 0.5870, 0.1140])
        f = (self.width / 2.0) / np.tan(np.pi * (self.fov / 2.0) / 180.0)
        disparity = self.sgm.computeDisparity(left, right)
        # show disparity map
        # cv2.imshow('disparity', disparity)
        # cv2.waitKey(1)
        disp = torch.tensor(disparity, dtype=torch.float32, device=self.device)
        # print("max: ", disp.max())
        # print("min: ", disp.min())
        depth = torch.where(disp > 0, f * self.baseline / disp, torch.tensor(0.0, device=self.device))
        # print(depth.max())
        return depth
        
