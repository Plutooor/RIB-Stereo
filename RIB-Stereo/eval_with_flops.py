import sys
sys.path.append('core_rt')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.RIB_stereo import RIBStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _StereoForward(torch.nn.Module):
    """Keep the same iteration count and inference branch for FLOPs tracing."""
    def __init__(self, model, iters):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, left, right):
        return self.model(left, right, iters=self.iters, test_mode=True)


class InferenceStats:
    """FPS counts stereo pairs; timing covers model forward only.

    FLOPs use fvcore's convention: one fused multiply-add is one FLOP.
    Only the first padded input, with batch size 1, is traced. Unsupported
    operators are reported; the result is an estimate, not an exact total.
    """
    def __init__(self, model, iters, mixed_prec, warmup=10):
        self.model = model
        self.iters = iters
        self.mixed_prec = mixed_prec
        self.warmup = max(0, warmup)
        self.warmed_shapes = set()
        self.profiled = False
        self.gflops = None
        self.flops_shape = None
        self.pairs = 0
        self.batches = 0
        self.seconds = 0.0

    def _forward(self, left, right):
        with autocast(enabled=self.mixed_prec):
            return self.model(left, right, iters=self.iters, test_mode=True)

    @staticmethod
    def _sync(tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    @torch.no_grad()
    def _profile(self, left, right):
        self.profiled = True
        self.flops_shape = tuple(left[:1].shape)
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            logging.warning('FLOPs unavailable: install fvcore with pip install fvcore. FPS is still measured.')
            return
        # The supplied script uses DataParallel(device_ids=[0]). Trace its
        # underlying model to avoid counting the wrapper's scatter/gather.
        net = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        wrapper = _StereoForward(net, self.iters).eval()
        try:
            with autocast(enabled=self.mixed_prec):
                analysis = FlopCountAnalysis(wrapper, (left[:1], right[:1]))
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                self.gflops = analysis.total() / 1e9
                unsupported = dict(analysis.unsupported_ops())
            print(f'FLOPs estimate: {self.gflops:.3f} GFLOPs/stereo pair; '
                  f'each input={self.flops_shape}, iters={self.iters}, '
                  f'AMP={self.mixed_prec}, 1 fused multiply-add = 1 FLOP')
            if unsupported:
                logging.warning('FLOPs excludes unsupported operators: %s', unsupported)
        except Exception as exc:
            self.gflops = None
            logging.warning('FLOPs tracing failed (%s). FPS is still measured.', exc)

    @torch.no_grad()
    def forward(self, left, right):
        if not self.profiled:
            self._profile(left, right)
        key = (tuple(left.shape), tuple(right.shape), left.dtype, right.dtype, left.device)
        if key not in self.warmed_shapes:
            for _ in range(self.warmup):
                self._forward(left, right)
            self.warmed_shapes.add(key)
        # CUDA launches are asynchronous, so synchronize at both boundaries.
        # Loading, transfers, padding, profiling, warmup, and metrics are excluded.
        self._sync(left)
        start = time.perf_counter()
        output = self._forward(left, right)
        self._sync(left)
        elapsed = time.perf_counter() - start
        self.seconds += elapsed
        self.pairs += int(left.shape[0])
        self.batches += 1
        return output, elapsed

    def summary(self, name):
        if self.pairs == 0:
            print(f'{name} performance: no samples measured')
            return {}
        fps = self.pairs / self.seconds
        batch_ms = 1000 * self.seconds / self.batches
        amortized_ms = 1000 * self.seconds / self.pairs
        flops_text = 'N/A' if self.gflops is None else f'{self.gflops:.3f} GFLOPs/pair (first input only)'
        print(f'{name} performance: FLOPs={flops_text}; FPS={fps:.2f} stereo pairs/s; '
              f'mean batch latency={batch_ms:.3f} ms; '
              f'amortized time/pair={amortized_ms:.3f} ms; '
              f'pairs={self.pairs}, batches={self.batches}, iters={self.iters}')
        return {f'{name}-fps': fps, f'{name}-batch-ms': batch_ms,
                f'{name}-ms-per-pair': amortized_ms,
                f'{name}-gflops-first-pair': self.gflops}


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, warmup=10):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    stats = InferenceStats(model, iters, mixed_prec, warmup)
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr, elapsed = stats.forward(image1, image2)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))

        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        # val = (valid_gt.flatten() >= 0.5)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1, **stats.summary('eth3d')}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, warmup=10, year=2012):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    stats = InferenceStats(model, iters, mixed_prec, warmup)
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training', year=2012)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr, elapsed = stats.forward(image1, image2)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(elapsed, '.3f')}s ({format(1/(elapsed), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}")
    return {'kitti-epe': epe, 'kitti-d1': d1, **stats.summary('kitti')}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, warmup=10, batch_size=8):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    stats = InferenceStats(model, iters, mixed_prec, warmup)
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
        pin_memory=True, shuffle=False, num_workers=8)

    out_list, epe_list = [], []
    for i_batch, (_, *data_blob) in enumerate(tqdm(val_loader)):
        image1, image2, disp_gt, valid_gt = [x for x in data_blob]

        image1 = image1.cuda()
        image2 = image2.cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        disp_pr, elapsed = stats.forward(image1, image2)
        disp_pr = padder.unpad(disp_pr).cpu()
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.abs(disp_pr - disp_gt)

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)
        if(np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test_sceneflow.txt', 'a')
    f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    print("Validation Scene Flow: %f, %f" % (epe, d1))
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1, **stats.summary('sceneflow')}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='MiddEval3', resolution='F', mixed_prec=False, warmup=10):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    stats = InferenceStats(model, iters, mixed_prec, warmup)
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, resolution=resolution)
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr, elapsed = stats.forward(image1, image2)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()
        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test_middlebury.txt', 'a')
    f.write("Validation Middlebury: %f, %f\n" % (epe, d1))

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1, **stats.summary('middlebury')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_rt/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='eth3d', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    parser.add_argument('--warmup', type=int, default=10, help='extra untimed warmup forwards for each input shape')
    parser.add_argument('--batch_size', type=int, default=8, help='Scene Flow validation batch size; use 1 for latency comparison')

    # Architecure choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=1, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, warmup=args.warmup)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, warmup=args.warmup)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, resolution=args.dataset[-1], mixed_prec=args.mixed_precision, warmup=args.warmup)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, warmup=args.warmup, batch_size=args.batch_size)