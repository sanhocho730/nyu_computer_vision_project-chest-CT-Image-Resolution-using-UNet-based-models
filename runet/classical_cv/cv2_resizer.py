import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from skimage import transform
import cv2
from common.loss import SSIM, PSNR, VGGPerceptualLoss


class Cv2ResizerEvaluation():
    def __init__(self, display_time=False):
        self.display_time = display_time

    def interpolate(self, img, output_size):
        start = time.time()
        interpolate_result = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        
        if self.display_time:
            print(f'Took {time.time() - start}s to compute a linear interpolation on an image of shape {img.shape}')
        
        return interpolate_result

    def evaluate_dataset(self, dataset, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        N = len(dataset)
        print(N)
        list_PSNR = np.zeros(N)
        list_SSIM = np.zeros(N)
        list_MSE = np.zeros(N)
        list_VGG = np.zeros(N)
        for i in tqdm(range(N)):
            sample= dataset[i]
            img = sample['image']
            label = sample['label']
            img_super = torch.FloatTensor(transform.resize(img, label.shape,order=1))
            list_PSNR[i] = 10 * torch.log10(torch.max(img_super)**2 / mse_loss(img_super, label))
            list_SSIM[i] = SSIM(img_super,label)
            list_MSE[i] = nn.MSELoss()(img_super,label)
            img_super = img_super[None,:,:,:].float().to(device)
            label = label.float()[None,:,:,:].to(device)
            list_VGG[i] = VGGPerceptualLoss()(img_super,label).cpu().detach().numpy()

        print("Mean PSNR of {:.03f} on test set with std of {:.03f}".format(np.mean(list_PSNR), np.std(list_PSNR)))
        print("Mean SSIM of {:.05f} on test set with std of {:.05f}".format(np.mean(list_SSIM), np.std(list_SSIM)))
        print("Mean MSE of {:.07f} on test set with std of {:.07f}".format(np.mean(list_MSE), np.std(list_MSE)))
        print("Mean VGG-Perceptual of {:.05f} on test set with std of {:.05f}".format(np.mean(list_VGG), np.std(list_VGG)))
