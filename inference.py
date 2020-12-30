import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

CHECK_SEG_ANN = True
THRESH = 0.01
large_count = 0
match_count = 0

def check_annotations(matte, df, full_path):
    bgr_colors = [
        # 0=background
        (0, 0, 0),
        # 15=person
        (128, 128, 192),
        # 12=dog
        (128, 0, 64),
        # 8=cat
        (0, 0, 64),
    ]
    global large_count, match_count
    row = df.loc[df["img_path"] == full_path]
    mask_path = row["seg_mask"].values[0]
    bgr_mask = cv2.imread(mask_path)
    mask = np.ndarray(shape=bgr_mask.shape[:2], dtype=np.uint8)
    mask[:, :] = 0
    idx = 1
    bgr_color = bgr_colors[idx]
    mask[(bgr_mask == bgr_color).all(2)] = 1.0
    ret = None
    max_val = matte.shape[0] * matte.shape[1] * 1.0
    # 1% of the image
    non_z_count = 0.05 * max_val
    if (np.count_nonzero(mask) >= non_z_count):
        large_count += 1
        seg_mask = cv2.resize(mask, (matte.shape[1], matte.shape[0]))
        seg_mask = seg_mask.astype(np.float32)
        masked = seg_mask * matte
        diff = cv2.absdiff(seg_mask, masked)
        sad = np.sum(diff)
        match_perc = sad / max_val
        print(match_perc)
        if (match_perc <= THRESH):
            match_count += 1
            ret = masked
    return ret

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images')
    parser.add_argument('--input-df', type=str, help='path of input images')
    parser.add_argument('--output-path', type=str, help='path of output images')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    # check input arguments
    if args.input_path:
        if not os.path.exists(args.input_path):
            print('Cannot find input path: {0}'.format(args.input_path))
            exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load(args.ckpt_path))
    modnet.eval()

    # inference images
    if args.input_path:
        im_names = os.listdir(args.input_path)
    elif args.input_df:
        import pandas as pd
        df = pd.read_csv(args.input_df, na_filter=False)
        if CHECK_SEG_ANN:
            df = df[df["seg_mask"] != ""]
        im_names = df.img_path.values
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        if args.input_df:
            full_path = im_name
            im = Image.open(full_path)
            im_name = os.path.basename(full_path)
        else:
            im = Image.open(os.path.join(args.input_path, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda(), True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        if args.input_df and CHECK_SEG_ANN:
            # check match percentage with segmentation annotation if available
            matte = check_annotations(matte, df, full_path)
        if matte is not None:
            matte = matte * 255
            matte_name = im_name.split('.')[0] + '.png'
            Image.fromarray(((matte).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
    print("{} images large enough, {} images accurate enough".format(large_count, match_count))
