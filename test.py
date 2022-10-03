from __future__ import print_function, division
import sys

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


def load_sequence(dir):
    imgs = os.listdir(dir)
    imgs = sorted(imgs)
    dict = {}

    first_frame = Image.open(osp.join(dir, imgs[0]))
    h, w = first_frame.size
    ff_np = np.array(first_frame)
    np.set_printoptions(threshold=np.inf)

    layers = np.max(ff_np)
    out = np.zeros((layers, len(imgs), w, h), dtype=np.uint8)
    for i in range(len(imgs)):
        current_frame = Image.open(osp.join(dir, imgs[i]))
        cf_np = np.array(current_frame)
        for j in range(layers):
            out[j, i, :, :] = (cf_np == (j + 1)).astype(np.uint8)

    return out, osp.join(dir, imgs[0])


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


def clip255(x):
    x = x * 1. / np.max(x)
    return np.clip((x * 255.).astype(np.uint8), 0, 255)


def seq_process(model, seq_dir):
    seq_np, ff_dir = load_sequence(seq_dir)

    seq_name = seq_dir.split('/')[-1]
    ckpt_name = args.restore_ckpt[5:-4]
    output_dir = osp.join('/home/hddb/hzm/Experiments/SPN/DAVIS2017', ckpt_name, seq_name)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    # print("Output path: {}".format(output_dir))
    copyfile(ff_dir, osp.join(output_dir, '00000.png'))
    objects, frames, w, h = np.shape(seq_np)
    # print('found {} objects in {} frames'.format(objects, frames))

    Transform = transforms.Compose([transforms.Resize([512, 896]), transforms.ToTensor()])
    iTransform = transforms.Resize([w, h])
    bitrates = []
    import time
    start = time.time()
    tt = 0.
    for i in range(1, frames):
        output = np.zeros((512, 896), dtype=np.uint8)
        bitrate = 0
        for j in range(objects):
            if np.all(seq_np[j, i, :, :] == 0):
                print('found no object {} in frame {}, skipped.'.format(j + 1, i))
                continue

            ref_img = (np.array(seq_np[j, i - 1, :, :]) * 1. / np.max(seq_np[j, i - 1, :, :]) * 255.).astype(np.uint8)
            input_img = (np.array(seq_np[j, i, :, :]) * 1. / np.max(seq_np[j, i, :, :]) * 255.).astype(np.uint8)

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

            flow_pred, _, _, bpp, _, t = model(input_3channel, ref_3channel)

            flow_pred_np = flow_pred[0][0].cpu().detach().numpy()
            flow_pred_cv2 = np.clip((flow_pred_np * 255.).astype(np.uint8), 0, 255)
            flow_pred_np_filtered = np.clip((np.around(flow_pred_cv2 / 255.) * 255.).astype(np.uint8), 0, 255)
            # cv2.imwrite(r'{}/{}_{}.png'.format(output_dir, i, j), flow_pred_np_filtered)
            # print("The bpp of object {} in frame {} is {}".format(j, i, bpp.data))
            bitrate += bpp.data
            mask = (flow_pred_np_filtered / 255).astype(np.uint8)
            unmask = np.bitwise_xor(mask, 1)
            output = output * unmask + (j + 1) * mask
            tt = tt + t

        output_pil = Image.fromarray(output.astype(np.uint8), mode='P')
        output_pil.putpalette((colormap).astype(np.uint8).flatten())
        output_pil = iTransform(output_pil)
        output_pil.save(r'{}/{:05d}.png'.format(output_dir, i))
        bitrates.append(bitrate)
    print(tt)
    end = time.time()
    print(end - start)
    # print("Mean bpp:{} = {} / {}".format(sum(bitrates) / frames, sum(bitrates), frames))
    return sum(bitrates) / frames, tt


def main():
    model = SFP(args)
    model.cuda()

    print("Loading checkpoint in {} ...".format(args.restore_ckpt))
    checkpoint = torch.load(args.restore_ckpt)
    current_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loading checkpoint success! {} epochs have been trained!".format(current_epoch))

    # seq_dir = r"/home/hddd/hzm/Dataset/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p/blackswan"
    # seqs_dir = r"/home/hddd/hzm/Dataset/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p_AOT"
    seqs_dir = r"/home/hddd/hzm/GitCode/STCN-main/davis_ori"
    seqs = os.listdir(seqs_dir)
    seqs = sorted(seqs)
    bpps = []
    ts = 0
    for i, seq in enumerate(seqs):
        seq_dir = osp.join(seqs_dir, seq)
        bpp, t = seq_process(model, seq_dir)
        bpps.append(bpp)
        print("{}/{} finished!".format(i + 1, len(seqs)))
        ts = ts + t
    print(ts)
    print(seqs)
    print(bpps)
    print("bpp = {} = {} / {}".format(sum(bpps) / len(bpps), sum(bpps), len(bpps)))


if __name__ == "__main__":
    main()
