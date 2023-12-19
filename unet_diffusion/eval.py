import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_image_subplot(image_path, title, subplot_num, num_subplots):
    img = mpimg.imread(image_path)
    plt.subplot(1, num_subplots, subplot_num)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    inf_names = list(glob.glob('{}/*_inf.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()
    inf_names.sort()

    psnr_ssim_pairs = []
    total_psnr = 0.0
    total_ssim = 0.0
    idx = 0
    for rname, fname, iname in zip(real_names, fake_names, inf_names):
        idx += 1
        ridx = os.path.basename(rname).rsplit('_hr', 1)[0]
        fidx = os.path.basename(fname).rsplit('_sr', 1)[0]
        assert ridx == fidx, 'Image ridx:{} != fidx:{}'.format(ridx, fidx)
    

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        inf_img = np.array(Image.open(iname))
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)

        total_psnr += psnr
        total_ssim += ssim
        psnr_ssim_pairs.append((psnr, ssim, rname, fname, iname))

    avg_psnr = total_psnr / idx
    avg_ssim = total_ssim / idx

    # Sorting based on PSNR and SSIM
    psnr_ssim_pairs.sort(key=lambda x: (-x[0], -x[1]))  # Sorting by PSNR then SSIM

    print(f'# Average Validation # PSNR: {avg_psnr:.4e}, SSIM: {avg_ssim:.4e}\n')
    print('# Top 3 Highest PSNR and SSIM:')
    for i, (psnr, ssim, rname, fname, iname) in enumerate(psnr_ssim_pairs[:3]):
        print(f"Rank {i+1}: PSNR: {psnr:.4e}, SSIM: {ssim:.4e}")
        print(f"HR Image: {rname}")
        print(f"SR Image: {fname}")
        print(f"INF Image: {iname}")


