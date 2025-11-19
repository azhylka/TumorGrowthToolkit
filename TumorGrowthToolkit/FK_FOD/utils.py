import itertools
import numpy as np
import os.path as osp
import nibabel as nib


def get_direction_to_index():
    mapping = {(x, y, z): (idx if idx < 13 else idx-1) for idx, (x, y, z) in enumerate(itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)))
                        if not (x == 0 and y == 0 and z == 0)}
    return mapping


def get_index_to_direction():
    mapping = {(idx if idx < 13 else idx-1): (x, y, z) for idx, (x, y, z) in enumerate(itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)))
                        if not (x == 0 and y == 0 and z == 0)}
    return mapping


def extract_dominant_discrete_orientation(discrete_fod_distribution):
    masking = np.sum(discrete_fod_distribution, axis=-1) > 0
    main_directions = np.argmax(discrete_fod_distribution, axis=-1)
    max_amplitudes = np.max(discrete_fod_distribution, axis=-1)
    
    index2direction = get_index_to_direction()
    lookup_table = np.stack([index2direction[idx] for idx in sorted(index2direction.keys())]) # skip the center direction (0,0,0)
    
    dominant_orientations = np.abs(lookup_table[main_directions].astype(np.float64))
    dominant_orientations[np.logical_not(masking)] = 0
    dominant_orientations *= max_amplitudes[..., np.newaxis]
    dominant_orientations /= np.max(dominant_orientations)
    return dominant_orientations


def create_fixel_volumes(fixel_directory, template_img_path):
    
    template_img = nib.load(template_img_path)
    
    # Load with stride handling (get_fdata applies strides automatically)
    fixel_amps = nib.load(osp.join(fixel_directory, 'fixel_amp.nii.gz')).get_fdata()
    directions_img = nib.load(osp.join(fixel_directory, 'directions.nii'))
    directions = directions_img.get_fdata()
    indices = nib.load(osp.join(fixel_directory, 'index.nii')).get_fdata()
    qform = directions_img.header.get_qform()
    for ax in range(3):
        if qform[ax, ax] < 0:
            directions = np.flip(directions, axis=ax)
            fixel_amps = np.flip(fixel_amps, axis=ax)

    max_fixels = int(np.max(indices[...,0]))
    for fixel_idx in range(max_fixels):
        mask = indices[...,0] >= fixel_idx+1

        fixel_volume = np.zeros(indices.shape[:-1]+(3,))
        amp_volume = np.zeros(indices.shape[:-1])
        fixel_indices = indices[mask, 1] + fixel_idx 
        fixel_volume[mask] = np.squeeze(directions[fixel_indices.astype(np.int64)])
        amp_volume[mask] = np.squeeze(fixel_amps[fixel_indices.astype(np.int64)])

        nib.save(nib.Nifti1Image(fixel_volume.astype(np.float32), affine=template_img.affine),
                 osp.join(fixel_directory, f'fixel_{fixel_idx}.nii.gz'))
        nib.save(nib.Nifti1Image(amp_volume.astype(np.float32), affine=template_img.affine),
                 osp.join(fixel_directory, f'amp_{fixel_idx}.nii.gz'))
        

def compute_fixel_ratios(fixel_directory):
    ratio_volume = None
    index_img = nib.load(osp.join(fixel_directory, 'index.nii'))
    indices = index_img.get_fdata() 
    ratio_volume = np.zeros(indices.shape[:-1]+(3,))

    for idx in np.unique(indices[...,0].astype(np.int8)):
        if idx == 0:
            continue
        idx -= 1
        ratio_volume[..., idx] = nib.load(osp.join(fixel_directory, f'amp_{int(idx)}.nii.gz')).get_fdata()

    total_amp = np.sum(ratio_volume, axis=-1)
    ratio_volume = np.divide(ratio_volume, total_amp[..., np.newaxis], where=total_amp[..., np.newaxis]!=0)

    nib.save(nib.Nifti1Image(ratio_volume.astype(np.float32), affine=index_img.affine),
             osp.join(fixel_directory, f'fixel_ratio_volume.nii.gz'))

        

def pair_closest_fixels(fixel_directory):
    """
    Pair closest fixels across multiple fixel volumes.
    
    Note: get_fdata() automatically applies strides from the NIfTI header,
    ensuring proper spatial scaling is accounted for during comparisons.
    """
    NEIGHBORHOOD_SIZE = 6
    index_img = nib.load(osp.join(fixel_directory, 'index.nii'))
    indices = index_img.get_fdata()

    fixel_amp_img = nib.load(osp.join(fixel_directory, 'fixel_amp.nii.gz'))

    for fixel_idx in range(3): # definetly not using more than 3 fixels
        fixel_img = nib.load(osp.join(fixel_directory, f'fixel_{fixel_idx}.nii.gz'))
        # get_fdata() applies strides automatically
        fixels = fixel_img.get_fdata()
        fixel_norms = np.linalg.norm(fixels, axis=-1)
        
        closest_fixels = np.zeros(fixels.shape[:-1] + (NEIGHBORHOOD_SIZE, 3))
        closest_fixel_amps = np.zeros(fixels.shape[:-1] + (NEIGHBORHOOD_SIZE,))
       
        for axis in range(3):
            for offset_idx, offset in enumerate([-1, 1]):
                deviations = []
                for other_fixel_idx in range(3):
                    if other_fixel_idx == fixel_idx:
                        other_fixels = np.copy(fixels)
                    else:
                        other_fixel_img = nib.load(osp.join(fixel_directory,
                                                            f'fixel_{other_fixel_idx}.nii.gz'))
                        other_fixels = other_fixel_img.get_fdata()

                    other_fixels = np.roll(other_fixels, shift=offset, axis=axis)
                    other_fixel_norm = np.linalg.norm(other_fixels, axis=-1)

                    fixel_order_mask = np.logical_and(indices[...,0] >= (other_fixel_idx+1), 
                                                      np.roll(indices[...,0], shift=offset, axis=axis) >= (other_fixel_idx+1))

                    dot_product = np.abs(np.sum(fixels[fixel_order_mask] * other_fixels[fixel_order_mask], axis=-1))
                    normalization = np.squeeze(fixel_norms[fixel_order_mask] * other_fixel_norm[fixel_order_mask])
                    dot_product = np.divide(dot_product, normalization,
                                            where=normalization > 0.00001)
                    deviation = np.zeros(fixel_order_mask.shape)
                    deviation[fixel_order_mask] = dot_product
                    deviations.append(deviation)

                closest_fixel_order = np.argmax(np.stack(deviations, axis=0), axis=0)

                for fixel_order in np.unique(closest_fixel_order):
                    amp_img = nib.load(osp.join(fixel_directory, f'amp_{fixel_order}.nii.gz')).get_fdata()

                    order_mask = np.logical_and(closest_fixel_order == fixel_order,
                                                np.logical_and(indices[...,0] >= (fixel_order+1),
                                                               indices[...,0] >= (fixel_idx+1)))
                    # if fixel_order == 0 and fixel_idx == 2:
                    #     nib.save(nib.Nifti1Image(order_mask.astype(np.float32), affine=fixel_img.affine),
                    #             osp.join(fixel_directory, f'order_mask.nii.gz'))
                        # return
                    if fixel_order == fixel_idx:
                        other_fixels = fixels
                    else:
                        other_fixel_img = nib.load(osp.join(fixel_directory,
                                                            f'fixel_{fixel_order}.nii.gz'))
                        other_fixels = other_fixel_img.get_fdata()
                    closest_fixels[order_mask, axis*2+offset_idx] = other_fixels[order_mask]
                    closest_fixel_amps[order_mask, axis*2+offset_idx] = amp_img[order_mask]
                    
        nib.save(nib.Nifti1Image(closest_fixels.astype(np.float32), affine=fixel_img.affine),
                osp.join(fixel_directory, f'closest_fixel_{fixel_idx}_to_fixels.nii.gz'))
        nib.save(nib.Nifti1Image(closest_fixel_amps.astype(np.float32), affine=fixel_img.affine),
                osp.join(fixel_directory, f'closest_fixel_{fixel_idx}_to_fixels_amps.nii.gz'))



def min_and_argmin_nibabel(img_volumes, out_min, out_argmin, use_nan_policy='ignore'):
    """
    Compute element-wise minimum and argmin across multiple volume arrays.
    
    Note: img_volumes should be pre-loaded with get_fdata() to ensure
    strides are already applied before passing to this function.
    
    Parameters
    ----------
    img_volumes : list of np.ndarray
        List of 3D or 4D volume arrays (typically from get_fdata())
    out_min : np.ndarray
        Minimum volume at each voxel
    out_argmin : np.ndarray
        Index of minimum at each voxel
    use_nan_policy : {'propagate', 'ignore'}
        How to handle NaN values
    """
    stacked = np.stack(img_volumes, axis=0)  # shape (3, X, Y, Z)

    if use_nan_policy == 'propagate':
        min_vol = np.min(stacked, axis=0)       # NaN propagates
        argmin_vol = np.argmin(stacked, axis=0) # argmin picks first occurrence on ties
    elif use_nan_policy == 'ignore':
        # np.nanmin and np.nanargmin behave like ignoring NaNs (but nanargmin fails if all NaN)
        min_vol = np.nanmin(stacked, axis=0)
        argmin_vol = np.nanargmin(stacked, axis=0)
    else:
        raise ValueError("use_nan_policy must be 'propagate' or 'ignore'")

    return min_vol, argmin_vol


if __name__ == "__main__":
    fixel_dir = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/fixels'
    template_img = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/T1w_acpc_dc_restore_1.25.nii.gz'
    # create_fixel_volumes(fixel_dir, template_img)
    # compute_fixel_ratios(fixel_dir)
    pair_closest_fixels(fixel_dir)
