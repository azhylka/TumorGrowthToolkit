#!/usr/bin/env python
"""
Combine 4D image volumes over the 4th dimension.

Each volume slice (along 4th dimension) is multiplied by its index+1, 
then all weighted volumes are summed to produce a 3D output.

Usage:
    python combine_5tt.py <input_4d_image> [output_image]

Example:
    python combine_5tt.py tissue_5tt.nii.gz tissue_combined.nii.gz
"""

import sys
import argparse
import numpy as np
import nibabel as nib


def combine_4d_volumes(img_4d, weights=None):
    """
    Combine 4D image volumes over the 4th dimension.
    
    Each volume slice is multiplied by its weight (default: index+1),
    then all weighted volumes are summed to produce a 3D output.
    
    Parameters
    ----------
    img_4d : np.ndarray
        4D numpy array of shape (X, Y, Z, N)
    weights : np.ndarray, optional
        1D array of shape (N,) with weights for each volume.
        Default: weights = np.arange(1, N+1) (i.e., [1, 2, 3, ...])
    
    Returns
    -------
    combined : np.ndarray
        3D combined volume of shape (X, Y, Z)
    """
    if img_4d.ndim != 4:
        raise ValueError(f"Expected 4D array, got {img_4d.ndim}D")
    
    n_volumes = img_4d.shape[3]
    
    if weights is None:
        weights = np.arange(1, n_volumes + 1, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (n_volumes,):
            raise ValueError(f"Weights shape {weights.shape} doesn't match 4th dimension {n_volumes}")
    
    # Apply weights to each volume slice and sum
    combined = np.zeros(img_4d.shape[:3], dtype=np.float32)
    for i in range(n_volumes):
        combined += img_4d[..., i] * weights[i]
    
    return combined


def decompose_3d_to_4d(img_3d, n_volumes, weights=None):
    """
    Decompose a 3D image back into a 4D image (inverse of combine_4d_volumes).
    
    Distributes the 3D volume across N volume slices, each divided by its weight 
    (default: index+1). This reverses the combination operation.
    
    Note: This is a pseudo-inverse and assumes the original 4D volumes were 
    identical or had a specific relationship. For perfect reconstruction, 
    you need additional information about the original distribution.
    
    Parameters
    ----------
    img_3d : np.ndarray
        3D numpy array of shape (X, Y, Z) to decompose
    n_volumes : int
        Number of volumes in the 4th dimension of output (N)
    weights : np.ndarray, optional
        1D array of shape (N,) with weights for each volume.
        Default: weights = np.arange(1, N+1) (i.e., [1, 2, 3, ...])
    
    Returns
    -------
    decomposed : np.ndarray
        4D array of shape (X, Y, Z, N) where each slice i is img_3d / weights[i]
    """
    if img_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {img_3d.ndim}D")
    
    if weights is None:
        weights = np.arange(1, n_volumes + 1, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (n_volumes,):
            raise ValueError(f"Weights shape {weights.shape} doesn't match n_volumes {n_volumes}")
    
    # Allocate output 4D array
    decomposed = np.zeros(img_3d.shape + (n_volumes,), dtype=np.float32)
    
    # Divide by weights to decompose into slices
    for i in range(n_volumes):
        if weights[i] != 0:
            decomposed[..., i] = img_3d == weights[i]
        else:
            decomposed[..., i] = 0
    
    return decomposed


def main():
    parser = argparse.ArgumentParser(
        description="Combine/decompose 4D image volumes over the 4th dimension"
    )
    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")
    
    # Combine subcommand
    combine_parser = subparsers.add_parser(
        "combine",
        help="Combine 4D image volumes into 3D (forward operation)"
    )
    combine_parser.add_argument(
        "input_image",
        type=str,
        help="Path to input 4D NIfTI image"
    )
    combine_parser.add_argument(
        "output_image",
        type=str,
        nargs="?",
        default=None,
        help="Path to output 3D combined image (default: input_base_combined.nii.gz)"
    )
    combine_parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Custom weights for each volume (default: 1, 2, 3, ...)"
    )
    combine_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    # Decompose subcommand
    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Decompose 3D image into 4D (inverse operation)"
    )
    decompose_parser.add_argument(
        "input_image",
        type=str,
        help="Path to input 3D NIfTI image"
    )
    decompose_parser.add_argument(
        "n_volumes",
        type=int,
        help="Number of volumes in the 4th dimension"
    )
    decompose_parser.add_argument(
        "output_image",
        type=str,
        nargs="?",
        default=None,
        help="Path to output 4D decomposed image (default: input_base_decomposed.nii.gz)"
    )
    decompose_parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Custom weights for each volume (default: 1, 2, 3, ...)"
    )
    decompose_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "combine":
        _combine_command(args)
    elif args.command == "decompose":
        _decompose_command(args)


def _combine_command(args):
    """Handle combine subcommand"""
    # Load input image
    if args.verbose:
        print(f"Loading 4D image: {args.input_image}")
    
    try:
        img = nib.load(args.input_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    if args.verbose:
        print(f"Input shape: {data.shape}")
        print(f"Input dtype: {data.dtype}")
    
    if data.ndim != 4:
        print(f"Error: Expected 4D image, got {data.ndim}D")
        sys.exit(1)
    
    # Parse weights if provided
    weights = None
    if args.weights is not None:
        weights = np.array(args.weights, dtype=np.float32)
        if args.verbose:
            print(f"Using custom weights: {weights}")
    else:
        # Default: weights = index + 1
        weights = np.arange(1, data.shape[3] + 1, dtype=np.float32)
        if args.verbose:
            print(f"Using default weights (index+1): {weights}")
    
    # Combine volumes
    if args.verbose:
        print("Combining volumes...")
    
    combined = combine_4d_volumes(data, weights=weights)
    
    if args.verbose:
        print(f"Combined shape: {combined.shape}")
        print(f"Combined dtype: {combined.dtype}")
        print(f"Combined value range: [{combined.min():.4f}, {combined.max():.4f}]")
    
    # Determine output path
    if args.output_image is None:
        base = args.input_image.rsplit(".", maxsplit=2 if args.input_image.endswith(".nii.gz") else 1)[0]
        args.output_image = f"{base}_combined.nii.gz"
    
    # Save output
    if args.verbose:
        print(f"Saving combined image to: {args.output_image}")
    
    try:
        out_img = nib.Nifti1Image(combined.astype(np.float32), affine=affine, header=header)
        nib.save(out_img, args.output_image)
        print(f"Success! Output saved to: {args.output_image}")
    except Exception as e:
        print(f"Error saving image: {e}")
        sys.exit(1)


def _decompose_command(args):
    """Handle decompose subcommand"""
    # Load input image
    if args.verbose:
        print(f"Loading 3D image: {args.input_image}")
    
    try:
        img = nib.load(args.input_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    if args.verbose:
        print(f"Input shape: {data.shape}")
        print(f"Input dtype: {data.dtype}")
    
    if data.ndim != 3:
        print(f"Error: Expected 3D image, got {data.ndim}D")
        sys.exit(1)
    
    # Parse weights if provided
    weights = None
    if args.weights is not None:
        weights = np.array(args.weights, dtype=np.float32)
        if args.verbose:
            print(f"Using custom weights: {weights}")
    else:
        # Default: weights = index + 1
        weights = np.arange(1, args.n_volumes + 1, dtype=np.float32)
        if args.verbose:
            print(f"Using default weights (index+1): {weights}")
    
    # Decompose volume
    if args.verbose:
        print(f"Decomposing into {args.n_volumes} volumes...")
    
    decomposed = decompose_3d_to_4d(data, args.n_volumes, weights=weights)
    
    if args.verbose:
        print(f"Decomposed shape: {decomposed.shape}")
        print(f"Decomposed dtype: {decomposed.dtype}")
        print(f"Decomposed value range: [{decomposed.min():.4f}, {decomposed.max():.4f}]")
    
    # Determine output path
    if args.output_image is None:
        base = args.input_image.rsplit(".", maxsplit=2 if args.input_image.endswith(".nii.gz") else 1)[0]
        args.output_image = f"{base}_decomposed.nii.gz"
    
    # Save output
    if args.verbose:
        print(f"Saving decomposed image to: {args.output_image}")
    
    try:
        # Update header for 4D output
        header.set_data_shape(decomposed.shape)
        out_img = nib.Nifti1Image(decomposed.astype(np.float32), affine=affine, header=header)
        nib.save(out_img, args.output_image)
        print(f"Success! Output saved to: {args.output_image}")
    except Exception as e:
        print(f"Error saving image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
