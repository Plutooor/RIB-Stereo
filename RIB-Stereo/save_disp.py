import sys
sys.path.append('core_rt')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core_rt.rt_igev_stereo import IGEVStereo
from core_rt.utils.utils import InputPadder
import core_rt.stereo_datasets as datasets
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
import matplotlib
matplotlib.use('Agg')


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def colorize_disp(disp, cmap='jet', vmin=0.0, vmax=192.0):
    """Colorize disparity on a fixed scale so different models use identical colors."""
    if vmax <= vmin:
        raise ValueError(f'color_vmax ({vmax}) must be greater than color_vmin ({vmin})')

    disp = disp.astype(np.float32)
    valid = np.isfinite(disp) & (disp > 0)
    rgb = np.zeros(disp.shape + (3,), np.uint8)
    if not np.any(valid):
        return rgb

    # Do not normalize with each image's own min/max: that makes colors incomparable.
    normed = np.clip((disp - vmin) / (vmax - vmin), 0, 1)
    rgb[valid] = (plt.get_cmap(cmap)(normed)[..., :3][valid] * 255).astype(np.uint8)
    return rgb


def save_disp(disp, file_stem, save_png=True, save_numpy=False, save_color=False, cmap='jet',
              color_vmin=0.0, color_vmax=192.0):
    """Save a disparity map (torch tensor [1,1,H,W] or [H,W]) to 16-bit png, npy and/or color vis."""
    file_stem = Path(file_stem)
    file_stem.parent.mkdir(parents=True, exist_ok=True)
    disp = disp.cpu().numpy().squeeze()
    if save_png:
        disp_16 = np.round(disp * 256).astype(np.uint16)
        skimage.io.imsave(str(file_stem) + '.png', disp_16)
    if save_numpy:
        np.save(str(file_stem) + '.npy', disp)
    if save_color:
        color_disp = colorize_disp(disp, cmap, color_vmin, color_vmax)
        Image.fromarray(color_disp).save(str(file_stem) + '_color.png')


@torch.no_grad()
def run_model(model, img1, img2, iters):
    img1 = img1[None].to(DEVICE)
    img2 = img2[None].to(DEVICE)
    padder = InputPadder(img1.shape, divis_by=32)
    img1, img2 = padder.pad(img1, img2)
    disp = model(img1, img2, iters=iters, test_mode=True)
    return padder.unpad(disp)


@torch.no_grad()
def save_sceneflow(model, args):
    """Save disparity for the SceneFlow test split (FlyingThings3D TEST),
    mirroring the folder structure of the dataset."""
    val_dataset = datasets.SceneFlowDatasets(aug_params=None, root=args.dataset_root,
                                             dstype='frames_finalpass', things_test=True)
    output_directory = Path(args.output_directory)
    print(f"Found {len(val_dataset)} images. Saving files to {output_directory}/")

    for val_id in tqdm(range(len(val_dataset))):
        imageL_file = val_dataset.image_list[val_id][0]
        _, img1, img2, _, _ = val_dataset[val_id]

        disp = run_model(model, img1, img2, args.valid_iters)

        # rel path like frames_finalpass/TEST/<scene>/<seq>/left/xxxx.png
        # -> keep <scene>/<seq>/xxxx to avoid name collisions
        rel = Path(imageL_file).relative_to(args.dataset_root)
        parts = rel.parts
        subdir = Path(*parts[2:-2]) if len(parts) > 3 else Path('.')
        file_stem = output_directory / subdir / Path(parts[-1]).stem
        save_disp(disp, file_stem, args.save_png, args.save_numpy, args.save_color, args.cmap,
                  args.color_vmin, args.color_vmax)


@torch.no_grad()
def save_eth3d(model, args):
    """Save disparity for the ETH3D two-view split (training by default),
    one folder per scene."""
    val_dataset = datasets.ETH3D(aug_params=None, root=args.dataset_root,
                                 split=args.eth3d_split)
    output_directory = Path(args.output_directory)
    print(f"Found {len(val_dataset)} images. Saving files to {output_directory}/")

    for val_id in tqdm(range(len(val_dataset))):
        imageL_file = val_dataset.image_list[val_id][0]
        _, img1, img2, _, _ = val_dataset[val_id]

        disp = run_model(model, img1, img2, args.valid_iters)

        # rel path like two_view_training/<scene>/im0.png -> <scene>/im0
        rel = Path(imageL_file).relative_to(args.dataset_root)
        parts = rel.parts
        subdir = Path(*parts[1:-1]) if len(parts) > 2 else Path('.')
        file_stem = output_directory / subdir / Path(parts[-1]).stem
        save_disp(disp, file_stem, args.save_png, args.save_numpy, args.save_color, args.cmap,
                  args.color_vmin, args.color_vmax)


@torch.no_grad()
def save_glob(model, args):
    """Original mode: infer disparity from a glob of left/right image pairs (e.g. KITTI)."""
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
        disp = padder.unpad(disp)
        file_stem = output_directory / Path(imfile1).name
        save_disp(disp, file_stem, args.save_png, args.save_numpy, args.save_color, args.cmap,
                  args.color_vmin, args.color_vmax)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    if args.dataset == 'sceneflow':
        save_sceneflow(model, args)
    elif args.dataset == 'eth3d':
        save_eth3d(model, args)
    elif args.dataset == 'glob':
        save_glob(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./checkpoints_rt/Sceneflow/rt-igev-stereo.pth')
    parser.add_argument('--dataset', help="which dataset to run on", default='sceneflow', choices=['sceneflow', 'eth3d', 'glob'])
    parser.add_argument('--dataset_root', help="root directory of the dataset", default='/media/ubuntu/7C4C5D504C5D0674/shengzhenshuo/data/SceneFlow')
    parser.add_argument('--eth3d_split', help="ETH3D two-view split", default='test', choices=['training', 'test'])
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as 16-bit png (disp * 256)')
    parser.add_argument('--no_save_png', action='store_true', help='disable 16-bit png output (use together with --save_color)')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--save_color', action='store_true', help='save color visualization (near = warm, far = cool)')
    parser.add_argument('--cmap', default='jet', help='colormap for --save_color (e.g. jet, turbo, plasma)')
    parser.add_argument('--color_vmin', type=float, default=0.0,
                        help='minimum disparity for colorization; keep it identical across models')
    parser.add_argument('--color_vmax', type=float, default=20,
                        help='maximum disparity for colorization; keep it identical across models')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames (only used with --dataset glob)", default="/media/ubuntu/7C4C5D504C5D0674/shengzhenshuo/data/KITTI-stereo/kitti_15/testing/image_2/*_10.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames (only used with --dataset glob)", default="/media/ubuntu/7C4C5D504C5D0674/shengzhenshuo/data/KITTI-stereo/kitti_15/testing/image_3/*_10.png")
    #parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/media/ubuntu/7C4C5D504C5D0674/shengzhenshuo/data/KITTI-stereo/kitti_12/testing/colored_0/*_10.png")
    #parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/media/ubuntu/7C4C5D504C5D0674/shengzhenshuo/data/KITTI-stereo/kitti_12/testing/colored_1/*_10.png")
    parser.add_argument('--output_directory', help="directory to save output", default="output/eth3d/test")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help='resolution of the disparity field (1/2^K)')
    parser.add_argument('--n_gru_layers', type=int, default=3, help='number of hidden GRU levels')
    parser.add_argument('--max_disp', type=int, default=192, help='max disp range')

    args = parser.parse_args()
    if args.no_save_png:
        args.save_png = False

    demo(args)
