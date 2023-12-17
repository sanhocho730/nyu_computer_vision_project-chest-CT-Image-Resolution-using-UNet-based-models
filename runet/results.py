import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import seaborn as sns
import pandas as pd

rc = {'axes.labelpad': 20,
      'axes.linewidth': 3,
      'axes.titlepad': 20,
      'axes.titleweight': "bold",
      'xtick.major.pad': 10,
      'ytick.major.pad': 10,
      'xtick.major.width': 3,
      'xtick.minor.width': 3,
      'ytick.major.width': 3,
      'xtick.major.size': 10,
      'xtick.minor.size': 6,
      'ytick.major.size': 10,
      'grid.linewidth': 3,
      'font.family': 'serif',
      'lines.linewidth': 2}

def set_rcParams(rc):
    if rc is None:
        pass
    else:
        for k, v in rc.items():
            plt.rcParams[k] = v


def save_fig(save_path):
    if save_path is not None:
        parent_path = Path(save_path)
        if not os.path.exists(parent_path.parent.absolute()):
            os.makedirs(parent_path.parent.absolute())
        plt.savefig(save_path, bbox_inches='tight', transparent=True)


def inverse_norm(image):
  if not isinstance(image, torch.Tensor):
    raise Exception('Input should be torch.Tensor!')
  # Tensor image to numpy
  image = image.cpu().permute(1, 2, 0).numpy()
  norm_mean = np.array([0.298, 0.298, 0.298])
  norm_std = np.array([0.223, 0.223, 0.223])
  image = (image * norm_std[None,None]) + norm_mean[None,None]
  image = np.clip(image, a_min=0.0, a_max=1.0)
  return image


def show_comparison(data, pred, label, titles=["Data", "Prediction", "Label"]):
  rc = {'axes.titlepad': 20,
        'axes.titleweight': "bold"}
  set_rcParams(rc)
  fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
  data = inverse_norm(data)
  pred = inverse_norm(pred.detach())
  label = inverse_norm(label)
  for i,(img, title) in enumerate(zip([data, pred, label], titles)):
    axes[i].imshow(img, cmap='gray')
    if titles is not None:
      axes[i].set_title(title)
    axes[i].set_axis_off()
  fig.tight_layout()
  plt.show()


def show_multiple_comparisons(label_set, data_set, pred_set, nn_pred_set, titles=["Ground truth", "Data", "RUNet\nprediction", "NN+RUNet\nprediction"], save_path=None):
  rc = {"axes.spines.left" : False,
        "axes.spines.right" : False,
        "axes.spines.bottom" : False,
        "axes.spines.top" : False,
        "xtick.bottom" : False,
        "xtick.labelbottom" : False,
        "ytick.labelleft" : False,
        "ytick.left" : False,
        'axes.titlepad': 12,
        'axes.titleweight': "bold",
        'axes.titlesize': 8, 
        "figure.subplot.wspace": 0.11,
        "figure.subplot.hspace": 0.08}
  set_rcParams(rc)
  fig, axes = plt.subplots(len(data_set), 4, figsize=(5,5), 
                           sharex=True, sharey=True, dpi=300)
  for i in range(len(data_set)):
    if i == 0:
      axes[i,0].set_title(titles[0])
      axes[i,1].set_title(titles[1])
      axes[i,2].set_title(titles[2])
      axes[i,3].set_title(titles[3])
    label = inverse_norm(label_set[i])
    data = inverse_norm(data_set[i])
    pred = inverse_norm(pred_set[i].detach())
    nn_pred = inverse_norm(nn_pred_set[i].detach())
    axes[i,0].imshow(label, cmap='gray')
    axes[i,1].imshow(data, cmap='gray')
    axes[i,2].imshow(pred, cmap='gray')
    axes[i,3].imshow(nn_pred, cmap='gray')

  save_fig(save_path)
  plt.show()

def get_ssim(data_set, pred_set, label_set, channel_axis=-1):
  ssim_preds = []
  ssim_baseline = []
  for i, (data, pred, label) in enumerate(zip(data_set, pred_set, label_set)):
    data = inverse_norm(data)
    pred = inverse_norm(pred.detach())
    label = inverse_norm(label)
    
    ssim_preds.append(ssim(pred, label, channel_axis=channel_axis))
    ssim_baseline.append(ssim(data, label, channel_axis=channel_axis))
  return ssim_preds, ssim_baseline


def get_psnr(data_set, pred_set, label_set):
  psnr_preds = []
  psnr_baseline = []
  for i, (data, pred, label) in enumerate(zip(data_set, pred_set, label_set)):
    data = inverse_norm(data)
    pred = inverse_norm(pred.detach())
    label = inverse_norm(label)
    
    psnr_preds.append(psnr(label, pred))
    psnr_baseline.append(psnr(label, data))
  return psnr_preds, psnr_baseline


def get_ssim_from_test_loader(test_loader, model):
  model.eval()
  ssim_preds = []
  ssim_baseline = []
  for batch_idx, img in enumerate(test_loader):
    print(f'batch {batch_idx}')
    data, label = img["image"], img["label"]
    with torch.no_grad():
      pred = model(data)
    batch_ssim_preds, batch_ssim_baseline = get_ssim(data, pred, label, channel_axis=-1)
    del data, label, pred
    ssim_preds += batch_ssim_preds
    ssim_baseline += batch_ssim_baseline
    del batch_ssim_preds, batch_ssim_baseline
  return ssim_preds, ssim_baseline

def get_psnr_from_test_loader(test_loader, model):
  model.eval()
  psnr_preds = []
  psnr_baseline = []
  for batch_idx, img in enumerate(test_loader):
    print(f'batch {batch_idx}')
    data, label = img["image"], img["label"]
    with torch.no_grad():
      pred = model(data)
    batch_psnr_preds, batch_psnr_baseline = get_psnr(data, pred, label)
    del data, label, pred
    psnr_preds += batch_psnr_preds
    psnr_baseline += batch_psnr_baseline
    del batch_psnr_preds, batch_psnr_baseline
  return psnr_preds, psnr_baseline
  

def draw_ssim_graph(ssim_baseline,ssim_preds,ssim_nn_preds, save_path=None):
  rc = {"axes.labelpad" : 12,
          "xtick.major.pad" : 12,
          "xtick.bottom" : True,
          "xtick.labelbottom" : True}
  sns.set_context("notebook", font_scale=1.2, rc=rc)
  df = pd.DataFrame({'Low-resolution image\n(baseline)': ssim_baseline,
                    'Restored image\nusing RUNet': ssim_preds,
                    'Restored image\nusing NN+RUNet': ssim_nn_preds})
  df = pd.melt(df, var_name='method type', value_name='SSIM')
  
  g = sns.catplot(df, x='method type', y='SSIM', 
                  kind='bar',
                  hue='method type',
                  dodge=False,
                  height=5,
                  width=0.3,
                  aspect=1.4, 
                  errorbar=("ci", 68),
                  lw=2)
  g.set_axis_labels("", "Structural similarity (SSIM)")            
  g.set(ylim=(0.6, 0.9))
  g.ax.margins(x=0.1)
  save_fig(save_path)
  plt.show()
  return df

def draw_psnr_graph(psnr_baseline, psnr_preds, psnr_nn_preds, save_path=None):
  rc = {"axes.labelpad" : 12,
        "xtick.major.pad" : 12}
  sns.set_context("notebook", font_scale=1.2, rc=rc)
  df = pd.DataFrame({'Low-resolution image\n(baseline)': psnr_baseline,
                     'Restored image\nusing RUNet': psnr_preds,
                     'Restored image\nusing NN+RUNet': psnr_nn_preds})
  df = pd.melt(df, var_name='method type', value_name='PSNR')
  
  g = sns.catplot(df, x='method type', y='PSNR', 
                  kind='bar',
                  hue='method type',
                  dodge=False,
                  height=5,
                  width=0.3,
                  aspect=1.4, 
                  errorbar=("ci", 68),
                  lw=2)
  g.set_axis_labels("", "Peak signal-to-noise ratio (dB)")            
  g.set(ylim=(10, 30))
  g.ax.margins(x=0.1)
  save_fig(save_path)
  plt.show()
  return df 











