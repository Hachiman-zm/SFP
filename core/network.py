import os
import math
import time

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from BitsEstimator import BitEstimator

from update import GMAUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from utils.flowlib import Enhance_Net
from utils.flowlib import torch_warp as warp
from gma import Attention, Aggregate
from Encoder import Analysis_mv_net
from Decoder import Synthesis_mv_net
from HyperEncoder import Analysis_prior_net
from HyperDecoder import Synthesis_prior_net
from ConditionalGaussianModel import ConditionalGaussianModel

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, test_mode=False, iters=12, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1) - 1.0
        image2 = 2 * (image2) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            cnet = cnet.detach()
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)
        coords0 = coords0.detach()
        coords1 = coords1.detach()

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up.detach()

            # flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


class SFP(nn.Module):
    def __init__(self, args, training=True):
        super(SFP, self).__init__()
        self.args = args
        self.semantic_flow_estimation = RAFTGMA(args)
        self.semantic_enhance_net = Enhance_Net()
        for p in self.parameters():
            p.requires_grad = False
        self.semantic_flow_encoder = Analysis_mv_net()
        self.semantic_flow_entropy = None
        self.semantic_flow_decoder = Synthesis_mv_net()
        self.semantic_mv_hyper_encoder = Analysis_prior_net()
        self.semantic_mv_hyper_decoder = Synthesis_prior_net()
        self.semantic_flow_warp = None
        self.conditional = ConditionalGaussianModel(1e-3, 1e-9)
        self.bitEstimator_mv = BitEstimator(channel=64)
        self.training = training

    def feature_probs_based_sigma(self, feature, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

        return total_bits, probs

    def iclr18_estrate_bits_mv(self, mv):
        prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def semantic_flow_enhancement(self, semantic_ref, semantic_mv):
        semantic_warp = warp(semantic_ref, semantic_mv)
        feature = torch.cat((semantic_warp[:, 0:1, :], semantic_ref[:, 0:1, :]), 1)
        enhancement = self.semantic_enhance_net(feature) + semantic_warp[:, 0:1, :]
        return enhancement, semantic_warp

    def forward(self, image2, image1, iters=12, flow_init=None, upsample=True, test_mode=False):
        shape = image2.shape
        t = 0.
        s1 = time.time()
        _, semantic_mv = self.semantic_flow_estimation(image1, image2, test_mode=True)
        if self.training == False:
            semantic_mv = semantic_mv.detach()
            _ = _.detach()
        unmasked_semantic_mv = semantic_mv
        if True:
            semantic_mv = semantic_mv * image1[:, 0:2, :, :]
        #t = t + time.time() - s1
        if self.training == True:
            semantic_mv_encoded = self.semantic_flow_encoder(semantic_mv)
        else:
            semantic_mv_encoded = self.semantic_flow_encoder(semantic_mv).detach()
        semantic_flow_noise = torch.nn.init.uniform_(torch.zeros_like(semantic_mv_encoded), -0.5, 0.5)

        if self.training == True:
            semantic_mv_encoded = semantic_mv_encoded + semantic_flow_noise
        else:
            semantic_mv_encoded = torch.round(semantic_mv_encoded)
            semantic_mv_encoded = semantic_mv_encoded.detach()
        if self.training == True:
            semantic_hyper_encoded = self.semantic_mv_hyper_encoder(semantic_mv_encoded)
        else:
            semantic_hyper_encoded = self.semantic_mv_hyper_encoder(semantic_mv_encoded).detach()
        semantic_hyper_noise = torch.nn.init.uniform_(torch.zeros_like(semantic_hyper_encoded), -0.5, 0.5)

        if self.training == True:
            semantic_hyper_encoded = semantic_hyper_encoded + semantic_hyper_noise
        else:
            semantic_hyper_encoded = torch.round(semantic_hyper_encoded)
            semantic_hyper_encoded = semantic_hyper_encoded.detach()
        # print('hyper_enc', semantic_hyper_encoded.shape)
        # print('mv_enc',semantic_mv_encoded.shape)
        # print('mv', semantic_mv.shape)

        s3 = time.time()
        if self.training == True:
            semantic_hyper_decoded = self.semantic_mv_hyper_decoder(semantic_hyper_encoded)
        else:
            semantic_hyper_decoded = self.semantic_mv_hyper_decoder(semantic_hyper_encoded).detach()

        if self.training == True:
            semantic_mv_decoded = self.semantic_flow_decoder(semantic_mv_encoded)
        else:
            semantic_mv_decoded = self.semantic_flow_decoder(semantic_mv_encoded).detach()

        total_bits_feature, _ = self.feature_probs_based_sigma(semantic_mv_encoded, semantic_hyper_decoded)
        s2 = time.time()
        semantic_flow, warped = self.semantic_flow_enhancement(image1, semantic_mv_decoded)
        if self.training == True:
            semantic_flow_clamp = torch.clamp(semantic_flow, 0, 1)
        else:
            semantic_flow_clamp = torch.clamp(semantic_flow, 0, 1).detach()
        t = t - s3 + time.time()

        pixels = shape[0] * shape[2] * shape[3]

        bpp_mv_hyper, _ = self.feature_probs_based_sigma(semantic_mv_encoded, semantic_hyper_decoded)
        if self.training == False:
            bpp_mv_hyper = bpp_mv_hyper.detach()
            _ = _.detach()

        bpp_mv, _ = self.iclr18_estrate_bits_mv(semantic_hyper_encoded)
        if self.training == False:
            bpp_mv = bpp_mv.detach()
            _ = _.detach()

        bpp_mv_hyper /= pixels
        bpp_mv /= pixels

        bpp = bpp_mv + bpp_mv_hyper

        '''
        total_bits_mv, _ = iclr18_estrate_bits_mv(semantic_mv_encoded)

        bpp = total_bits_mv / shape[0] / shape[2] / shape[3]
        '''

        return semantic_flow_clamp, semantic_mv.cpu().detach().numpy(), warped, bpp, unmasked_semantic_mv.cpu().detach().numpy(), t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

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
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    args = parser.parse_args()

    test_model = SFP(args).cuda()

    test1 = torch.zeros((1, 3, 256, 256)).cuda()
    test2 = torch.ones((1, 3, 256, 256)).cuda()
    out, _, _, out2 = test_model(test1, test2)
    # print(out)
    print(type(out))
    print(out2)
