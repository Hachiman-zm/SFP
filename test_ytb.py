"""
YouTubeVOS has a label structure that is more complicated than DAVIS
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set
Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

the statement above comes from https://github.com/hkchengrex/STCN/blob/main/eval_youtube.py, which I think it is useful to understand dataset YTB-VOS
"""
from __future__ import print_function, division
import sys

import math
import numpy as np

sys.path.append('core')
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--output', type=str, default='checkpoints',
                    help='output directory to save checkpoints and plots')
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--upsample-learn', action='store_true', default=False,
                    help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--val_freq', type=int, default=10000,
                    help='validation frequency')
parser.add_argument('--mixed_precision', default=False, action='store_true',
                    help='use mixed precision')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--Lambda', type=float, default=0.00001)
parser.add_argument('--model_name', default='', help='specify model name')

parser.add_argument('--position_only', default=False, action='store_true',
                    help='only use position-wise attention')
parser.add_argument('--position_and_content', default=False, action='store_true',
                    help='use position and content-wise attention')
parser.add_argument('--num_heads', default=1, type=int,
                    help='number of heads in attention and aggregation')
parser.add_argument('--max-epochs', default=100000, type=int)
parser.add_argument('--gpu_id', type=str, required=True)
parser.add_argument('--loss', type=str, default='L2')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

import torch
import torch.nn as nn
from torchvision import transforms

import os.path as osp

import cv2
from PIL import Image
from shutil import copyfile

from core.network import SFP
from core.utils import flow_viz


def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32)
    return cmap


colormap = label_colormap(255)


def load_seq(seq_dir, ref_dir):
    '''
        preprocess ref_frame
        '''
    refs = os.listdir(ref_dir)
    refs = sorted(refs)
    start_frame = int(refs[0].split('.')[0])
    ref0_pil = Image.open(osp.join(ref_dir, refs[0]))
    w, h = ref0_pil.size
    index_set = set()
    start_dict = {}
    max_idx = 0

    for index, ref in enumerate(refs):
        ref_img = osp.join(ref_dir, ref)
        ref_pil = Image.open(ref_img)
        ref_np = np.array(ref_pil)
        for item in ref_np.flat:
            if item not in index_set:
                index_set.add(item)
                start_dict[str(item)] = int(ref.split('.')[0]) - start_frame
                max_idx = max(max_idx, item)
    start_frame = np.zeros(max_idx + 1)
    for key in start_dict.keys():
        start_frame[int(key)] = int(start_dict[key])

    imgs = os.listdir(seq_dir)
    annotations = np.zeros((max_idx + 1, len(imgs), h, w))
    flags = np.zeros((max_idx + 1, len(imgs)))
    # 0 no object
    # 1 with object
    # 2 start frame
    # 3 end frame

    imgs = sorted(imgs)
    for i, img in enumerate(imgs):
        img_dir = osp.join(seq_dir, img)
        img_dir = Image.open(img_dir)
        img_np = np.array(img_dir)
        for j in range(1, max_idx + 1):
            annotations[j, i, :, :] = (img_np == j).astype(np.uint8)
            flags[j, i] = (img_np == j).any().astype(np.uint8)

    for j in range(1, max_idx + 1):
        flags[j, int(start_frame[j])] = 2
        for k in range(len(imgs)):
            tk = len(imgs) - k - 1
            if flags[j, tk] == 1:
                flags[j, tk] = 3
                break
    return annotations, flags, osp.join(ref_dir, refs[0])


def seq_process(model, seq_dir, ref_dir):
    '''
    :param model:
    :param seq_dir: annotations with all frames
    :param ref_dir: valid with first frame
    :return:
    '''

    ann_np, flg_np, ff_dir = load_seq(seq_dir, ref_dir)

    seq_name = seq_dir.split('/')[-1]
    ckpt_name = args.restore_ckpt[5:-4]
    output_dir = osp.join('/home/hddb/hzm/Experiments/SPN/ytb_vos2019/STCN', ckpt_name, seq_name)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    print("Output path: {}".format(output_dir))
    copyfile(ff_dir, osp.join(output_dir, '00000.png'))
    objects, frames, h, w = np.shape(ann_np)
    print('found {} objects in {} frames with hxw={}x{}'.format(objects, frames, h, w))
    h_new = math.ceil(h / 64) * 64
    w_new = math.ceil(w / 64) * 64
    if h_new > 1280:
        h_new = 1280
    if w_new > 1280:
        w_new = 1280
    Transform = transforms.Compose([transforms.Resize([h_new, w_new]), transforms.ToTensor()])
    iTransform = transforms.Resize([h, w])
    bitrates = []
    for i in range(1, frames):
        # output = np.zeros((512, 896), dtype=np.uint8)
        output = np.zeros((h_new, w_new), dtype=np.uint8)
        bitrate = 0
        for j in range(1, objects):
            if flg_np[j, i] == 0:
                # print('found no object {} in frame {}, skipped.'.format(j, i))
                continue
            if flg_np[j, i] == 2:
                # print('object {} is the reference of  frame {}, skipped.'.format(j, i))
                continue

            ref_img = (np.array(ann_np[j, i - 1, :, :]) * 1. / np.max(ann_np[j, i - 1, :, :]) * 255.).astype(np.uint8)
            input_img = (np.array(ann_np[j, i, :, :]) * 1. / np.max(ann_np[j, i, :, :]) * 255.).astype(np.uint8)

            ref_img = Image.fromarray(ref_img)
            input_img = Image.fromarray(input_img)

            ref_torch = Transform(ref_img)
            input_torch = Transform(input_img)

            ref_torch = ref_torch.unsqueeze(0)
            input_torch = input_torch.unsqueeze(0)

            ref_3channel = torch.cat((ref_torch, ref_torch, ref_torch), 1)
            input_3channel = torch.cat((input_torch, input_torch, input_torch), 1)

            ref_3channel = ref_3channel.cuda()
            input_3channel = input_3channel.cuda()

            flow_pred, _, _, bpp, _ = model(input_3channel, ref_3channel)

            flow_pred_np = flow_pred[0][0].cpu().detach().numpy()
            flow_pred_cv2 = np.clip((flow_pred_np * 255.).astype(np.uint8), 0, 255)
            flow_pred_np_filtered = np.clip((np.around(flow_pred_cv2 / 255.) * 255.).astype(np.uint8), 0, 255)
            # cv2.imwrite(r'{}/{}_{}.png'.format(output_dir, i, j), flow_pred_np_filtered)
            # print("The bpp of object {} in frame {} is {}".format(j, i, bpp.data))
            bitrate += bpp.data
            mask = (flow_pred_np_filtered / 255).astype(np.uint8)
            unmask = np.bitwise_xor(mask, 1)
            output = output * unmask + j * mask

        output_pil = Image.fromarray(output.astype(np.uint8), mode='P')
        output_pil.putpalette((colormap).astype(np.uint8).flatten())
        output_pil = iTransform(output_pil)
        output_pil.save(r'{}/{:05d}.png'.format(output_dir, i))
        bitrates.append(bitrate)
    return sum(bitrates) / frames, frames


def test_YTB():
    model = SFP(args, training=False)
    model.cuda()

    print("Loading checkpoint in {} ...".format(args.restore_ckpt))
    checkpoint = torch.load(args.restore_ckpt)
    current_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loading checkpoint success! {} epochs have been trained!".format(current_epoch))

    # seq_root = r"/home/hddd/hzm/Dataset/VOS-2019/valid_all_frames/Masks/AOT"
    seq_root = r"/home/hddd/hzm/GitCode/STCN-main/ytb_ori_all/Annotations"
    ref_root = r"/home/hddd/hzm/Dataset/VOS-2019/valid/Annotations"
    seqs = os.listdir(seq_root)
    seqs = sorted(seqs)
    bpps = []
    for i, seq in enumerate(seqs):
        seq_dir = osp.join(seq_root, seq)
        ref_dir = osp.join(ref_root, seq)
        ckpt_name = args.restore_ckpt[5:-4]
        bpp_output_dir = osp.join('/home/hddb/hzm/Experiments/SPN/ytb_vos2019/STCN', ckpt_name, 'bpp')
        if not osp.exists(bpp_output_dir):
            os.makedirs(bpp_output_dir)
        if osp.exists(osp.join(bpp_output_dir, "{}.log".format(seq))):
            with open(osp.join(bpp_output_dir, "{}.log".format(seq)), 'r') as f:
                lines = f.readlines()
                bpp, frames = lines[0].split()
                bpp = float(bpp)
                frames = int(frames)
        else:
            bpp, frames = seq_process(model, seq_dir, ref_dir)
            with open(osp.join(bpp_output_dir, "{}.log".format(seq)), 'w') as f:
                f.write("{} {}".format(bpp, frames))
        bpps.append(bpp)
        print("{}/{} finished!".format(i + 1, len(seqs)))

    print("bpp = {} = {} / {}".format(sum(bpps) / len(bpps), sum(bpps), len(bpps)))


if __name__ == "__main__":
    test_YTB()
