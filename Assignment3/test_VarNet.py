"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
from evaluation_metrics import ssim, nmse, mse, psnr
import h5py

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import glob

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data.transforms import center_crop


from fastmri.pl_modules import FastMriDataModule, VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations,
    )
    # mask = create_mask_for_mask_type(
    #     12, args.center_fractions, args.accelerations,
    # )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform(mask_func=mask)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # print how large the training/validation set is
    test_set = data_module.test_dataloader().dataset
    print("Size of trainingset:", len(test_set))


    # ------------
    # model
    # ------------

    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        interpolation_method=args.interpolation_method,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    return


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("save_model/fastmri_dirs.yaml")
    #path_config = pathlib.Path("/content/gdrive/MyDrive/DL_4_MI/Assigment3/save_model/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 1

    # set defaults based on optional directory config
    data_path = "/gpfs/work5/0/prjs1312/Recon_exercise/FastMRIdata/"
    default_root_dir = fetch_dir("log_path", path_config) / "varnet" / "varnet_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="test",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[1.5],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs to train",
    )

    parser.add_argument(
        "--interpolation_method",
        default="nearest",
        choices=("nearest", "fourier", "bspline", "rbf"),
        type=str,
        help="Interpolation method for missing k-space values",
    )
    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        # mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        mask_type="random",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,
    )

    # module config
    parser = VarNetModule.add_model_specific_args(parser)
    args = parser.parse_args()
    parser.set_defaults(
        num_cascades=2,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=args.learning_rate,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        # strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TESTING
    # ---------------------
    cli_main(args)


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]



def evaluate_test_data_quantitatively(datapath, reconpath):
    #######################
    # Start YOUR CODE    #
    #######################

    # load in ground truth and reconstruction images

    # NOTE: Reconstructed image is cropped by the VarNet
    # the ground truth image still needs to be cropped 
    # Use: gt = center_crop(gt, recon.shape)
    metrics = {"MSE": [], "NMSE": [], "PSNR": [], "SSIM": []}
    # quantitative evaluation
    gt_files = sorted(glob.glob(os.path.join(datapath, "*.h5")))
    recon_files = sorted(glob.glob(os.path.join(reconpath, "*.h5")))

    # Create a dictionary to match recon files by filename
    recon_dict = {os.path.basename(f): f for f in recon_files}

    for gt_file in gt_files:
        fname = os.path.basename(gt_file)
        if fname not in recon_dict:
            print(f"Warning: No corresponding reconstruction found for {fname}")
            continue  # Skip unmatched files

        recon_file = recon_dict[fname]

        with h5py.File(gt_file, "r") as f_gt, h5py.File(recon_file, "r") as f_recon:
            kspace = f_gt["kspace"][:]  # Load k-space
            gt = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace)))
            recon = f_recon["reconstruction"][:]  
            gt = np.squeeze(gt)

            # Ensure the crop is applied only to height and width dimensions
            gt = center_crop(gt, recon.shape[1:])
            # Compute metrics
            metrics["MSE"].append(mse(gt, recon))
            metrics["NMSE"].append(nmse(gt, recon))
            metrics["PSNR"].append(psnr(np.abs(gt), np.abs(recon)))
            metrics["SSIM"].append(ssim(np.abs(gt), np.abs(recon)))

    # Compute mean and std for each metric
    avg_metrics = {key: np.mean(values) if values else float('nan') for key, values in metrics.items()}
    std_metrics = {key: np.std(values) if values else float('nan') for key, values in metrics.items()}

    # Print results
    print("Evaluation Results:")
    for key in metrics.keys():
        print(f"{key}: Mean = {avg_metrics[key]:.4f}, Std = {std_metrics[key]:.4f}")


    #######################
    # END OF YOUR CODE    #
    #######################
    return


def evaluate_test_data_qualitatively(datapath, reconpath, save_dir="qualitative_results_rbf4"):
    #######################
    # Start YOUR CODE    #
    #######################

    # load in ground truth and reconstruction images

    # NOTE: Reconstructed image is cropped by the VarNet
    # the ground truth image still needs to be cropped 
    # Use: gt = center_crop(gt, recon.shape)

    # qualitative evaluation
    os.makedirs(save_dir, exist_ok=True)

    # Get sorted lists of files
    gt_files = sorted(glob.glob(os.path.join(datapath, "*.h5")))
    recon_files = sorted(glob.glob(os.path.join(reconpath, "*.h5")))

    # Create a dictionary to match recon files by filename
    recon_dict = {os.path.basename(f): f for f in recon_files}

    for gt_file in gt_files:
        fname = os.path.basename(gt_file)
        if fname not in recon_dict:
            print(f"Warning: No corresponding reconstruction found for {fname}")
            continue  # Skip unmatched files

        recon_file = recon_dict[fname]

        with h5py.File(gt_file, "r") as f_gt, h5py.File(recon_file, "r") as f_recon:
            print(f"Evaluating {fname}...")

            # Load ground truth and reconstruction
            kspace = f_gt["kspace"][:]  # Load k-space
            gt = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace)))
            recon = f_recon["reconstruction"][:]
            gt = center_crop(gt, recon.shape[1:])  # Crop ground truth to match reconstruction size

            # Select center slice
            center_slice_idx = gt.shape[0] // 2
            gt_slice = gt[center_slice_idx]
            recon_slice = recon[center_slice_idx]

            gt_slice = np.squeeze(gt[center_slice_idx])

            # Compute magnitude, phase, real, imaginary
            gt_mag, gt_phase, gt_real, gt_imag = np.abs(gt_slice), np.angle(gt_slice), np.real(gt_slice), np.imag(gt_slice)
            recon_mag, recon_phase, recon_real, recon_imag = np.abs(recon_slice), np.angle(recon_slice), np.real(recon_slice), np.imag(recon_slice)
            
            # Store all components
            gt_components = [gt_mag, gt_phase, gt_real, gt_imag]
            recon_components = [recon_mag, recon_phase, recon_real, recon_imag]
            titles = ["Magnitude", "Phase", "Real", "Imaginary"]
            cmaps = ["gray", "gray", "gray", "gray"]  

            # Plot results
            fig, axes = plt.subplots(4, 2, figsize=(10, 12))

            for j in range(4):
                axes[j, 0].imshow(gt_components[j], cmap=cmaps[j])
                axes[j, 0].set_title(f"GT {titles[j]}")
                axes[j, 1].imshow(recon_components[j], cmap=cmaps[j])
                axes[j, 1].set_title(f"Reconstructed {titles[j]}")

                for ax in axes[j]:
                    ax.axis("off")

            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(save_dir, f"{fname.replace('.h5', '.png')}")
            plt.savefig(save_path, dpi=300)

            plt.close()  # Close figure to free memory

    return


if __name__ == "__main__":
    # run testing the network
    run_cli()
    datapath = '/gpfs/work5/0/prjs1312/Recon_exercise/FastMRIdata/multicoil_test/'
    reconpath = 'varnet/varnet_demo/reconstructions/'
    # # quantitativaly evaluate data
    evaluate_test_data_quantitatively(datapath, reconpath)
    # # qualitatively
    evaluate_test_data_qualitatively(datapath, reconpath)
