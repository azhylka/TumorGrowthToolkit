import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
from contextlib import contextmanager

def segment_direction(vec, segment_center):
    cos_dist = segment_center @ vec.T 
    segment = np.argmax(cos_dist, axis=0)  # which of the 26 is closest
    return segment  # index of segment (0–25)

def average_segment(segment_vectors, amplitudes):
    segment_vectors = np.repeat(np.expand_dims(segment_vectors, axis=0), amplitudes.shape[0], axis=0)
    segment_vectors *= amplitudes[..., None]
    mean_vector = np.sum(segment_vectors, axis=1) / segment_vectors.shape[1]
    # mean_vector_amplitude = np.linalg.norm(mean_vector, axis=1)

    return mean_vector


def neighborhood_directions(sphere_amps, dirs, dir_segment):
    segments = np.sort(np.unique(dir_segment))
    segment_orientations = np.zeros((sphere_amps.shape[0], len(segments), 3))
    for idx, segment in enumerate(segments):
        segment_mask = dir_segment == segment
        seg_dirs = dirs[segment_mask]

        mean_vec = average_segment(seg_dirs, sphere_amps[:,segment_mask])
        segment_orientations[:,idx,:] = mean_vec
    return segment_orientations

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into a tqdm progress bar."""
    class BatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = BatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


if __name__ == "__main__":

    print("Loading data...")
    # sphere_amps_img = nib.load('HCP1065_FOD_sphere362_space-sri.nii.gz')
    sphere_amps_img = nib.load('/Users/azhylka/Projects/TUMor_Data/HCP/100307/wmfod_norm_sphere362.nii.gz')
    print(f"Reshaping data into {(-1,)+ sphere_amps_img.shape[-2:]}...")
    sphere_amps = sphere_amps_img.get_fdata().reshape((-1,)+ sphere_amps_img.shape[-2:])

    print("Loading directions...")
    dirs = np.loadtxt('/Users/azhylka/Projects/TumorGrowthToolkit/TumorGrowthToolkit/FK_FOD/sphere362.txt')
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]

    # all combinations of -1, 0, 1 except (0,0,0)
    dirs_26 = np.array([[x, y, z] 
                        for x in (-1, 0, 1)
                        for y in (-1, 0, 1)
                        for z in (-1, 0, 1)
                        if not (x == 0 and y == 0 and z == 0)], dtype=float)

    # normalize to unit vectors
    dirs_26 /= np.linalg.norm(dirs_26, axis=1)[:, None]

    dir_segment = segment_direction(dirs, dirs_26)

    # neighborhood_directions(np.squeeze(sphere_amps[:,50,:]), dirs, dir_segment)
    
    total_tasks = sphere_amps.shape[1]
    print(f"Processing {sphere_amps.shape[0]} x {total_tasks} voxels...")
    with tqdm_joblib(tqdm(total=total_tasks)) as progress:
        odf_26dir = Parallel(n_jobs=-4)(
            delayed(neighborhood_directions)(np.squeeze(sphere_amps[:,i,:]), dirs, dir_segment)
            for i in range(total_tasks)
        )
    
    odf_26dir = list(map(lambda x: np.expand_dims(np.reshape(x, sphere_amps_img.shape[:2]+x.shape[-2:]),axis=2), odf_26dir))
    odf_26dir = np.concatenate(odf_26dir, axis=2)
    nib.save(nib.Nifti1Image(odf_26dir, affine=sphere_amps_img.affine), 
             '/Users/azhylka/Projects/TUMor_Data/HCP/100307/wmfod_norm_sphere362_segmented.nii.gz')
            #   'HCP1065_FOD_sphere362_segmented.nii.gz')
    amplitudes = np.linalg.norm(odf_26dir, axis=-1)
    nib.save(nib.Nifti1Image(amplitudes, affine=sphere_amps_img.affine), 
             '/Users/azhylka/Projects/TUMor_Data/HCP/100307/wmfod_norm_sphere362_segmented_amps.nii.gz')
            #  'HCP1065_FOD_sphere362_segmented_amps.nii.gz')
    
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # segment_orientations *= segment_amplitudes
    # for i in range(len(segment_orientations)):
    #     ax.plot([0, segment_orientations[i,0]], 
    #             [0, segment_orientations[i,1]], 
    #             [0, segment_orientations[i,2]])
    # ax.scatter(segment_orientations[:,0], segment_orientations[:,1], segment_orientations[:,2],
    #             c=segments, cmap='tab10', s=10)

    # plt.show()
