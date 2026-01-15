import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
import os

from basicsr.utils import img2tensor, scandir

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


import torch
def main(args):
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = args.gt
    folder_restored = args.restored
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))
    ###################
    img_list.sort(key=lambda x:int(x.split('.')[0].split('/')[-1]))
    ###################

    img_list_restored = sorted(list(scandir(folder_restored, recursive=True, full_path=True)))
    ###################
    img_list_restored.sort(key=lambda x:int(x.split('x2_Ours')[0].split('/')[-1]))
    ###################

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_path_restored = img_list_restored[i]
        print(img_list_restored[i])
        print(img_path)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        with torch.no_grad():
            # calculate lpips
            lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
            print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val[0][0][0][0]:.6f}.')
            lpips_all.append(lpips_val[0][0][0][0])

    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='/output/datasets/dehaze_derain/benchmark/RainDS/test_syn/HR/', help='Path to gt')
    parser.add_argument('--restored', type=str, default='/home/IR/results/Ours_rainstreak_raindrop_true_105k/visualization/rainstreak_raindrop/', help='Path to restored')
    args = parser.parse_args()
    main(args)

    # /home/IR/results/Ours_rainstreak_/visualization/rainstreak/  /output/datasets/dehaze_derain/benchmark/RainDS/test_syn/HR/ /output/datasets/dehaze_derain/benchmark/Raindrop/HR/
