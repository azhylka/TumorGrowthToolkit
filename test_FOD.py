from TumorGrowthToolkit.FK_FOD import FK_FOD_Solver
from TumorGrowthToolkit.FK_DTI import FK_DTI_Solver
from TumorGrowthToolkit.FK import Solver as FK_Solver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib
import os
from scipy.ndimage import binary_dilation
import TumorGrowthToolkit.FK_DTI.tools as tools


def get_FK_solver_solution(x, y, z, RatioDw_Dg, affine, dw, rho, gm, wm, init_scale, resolution_factor, stoppingVolume):

    parametersFK = {
        'Dw': dw,          # maximum diffusion coefficient
        'rho': rho,        # Proliferation rate
        'gm' : gm,
        'wm' : wm,
        'NxT1_pct': x,    # tumor position [%]
        'NyT1_pct': y,
        'NzT1_pct': z,
        'init_scale': init_scale, #scale of the initial gaussian
        'resolution_factor': resolution_factor, #resultion scaling for calculations
        'verbose': True, #printing timesteps 
        'time_series_solution_Nt': 64, # number of timesteps in the output
        'stopping_volume': stoppingVolume,
        'stopping_time': 1000,
        'use_homogen_gm': True,
        'RatioDw_Dg': RatioDw_Dg
    }

    fkSolver = FK_Solver(parametersFK)
    resultFK = fkSolver.solve()
    nib.save(nib.Nifti1Image(resultFK['final_state'].astype(np.float32), affine), f'./dataset/FK_result_{RatioDw_Dg}.nii.gz')
    return resultFK

def get_FK_DTI_solution(x, y, z, RatioDw_Dg, affine, dw, rho, gm, wm, init_scale, resolution_factor, stoppingVolume):
    tensorPath = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/dt.nii.gz'
    tissueTensor = tools.get_tensor_from_lower6_mrtrix(nib.load(tensorPath).get_fdata()[:,:,:,:])
    tissueTensor[np.isnan(tissueTensor)] = 0

    parameters = {
        'Dw': dw,          # maximum diffusion coefficient
        'rho': rho,        # Proliferation rate
        'gm' : gm,
        'wm' : wm,
        'diffusionTensors': tissueTensor, # diffusion tissue map as shown above
        'diffusionTensorExponent': 1, # exponent for the diffusion tensor, 1.0 for linear relationship
        'diffusionEllipsoidScaling':1,#21.713178343886213,
        'NxT1_pct': x,    # tumor position [%]
        'NyT1_pct': y,
        'NzT1_pct': z,
        'init_scale': init_scale, #scale of the initial gaussian
        'resolution_factor': resolution_factor, #resultion scaling for calculations
        'verbose': False, #printing timesteps 
        'time_series_solution_Nt': 64, # number of timesteps in the output
        'stopping_volume': stoppingVolume,
        'stopping_time': 1000,
        'use_homogen_gm': True,
        'RatioDw_Dg': RatioDw_Dg
    }

    dtiSolver = FK_DTI_Solver(parameters)
    resultDTI = dtiSolver.solve()
    nib.save(nib.Nifti1Image(resultDTI['final_state'].astype(np.float32), affine), f'./dataset/FK_DTI_result_{RatioDw_Dg}.nii.gz')
    return resultDTI

def get_FK_FOD_solution(x, y, z, RatioDw_Dg, affine, dw, rho, gm, wm, init_scale, resolution_factor, stoppingVolume):
    fod_26dir_file = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/wmfod_norm_sphere362_segmented_amps.nii.gz'

    fod_disrtibution = nib.load(fod_26dir_file).get_fdata()

    amp_sums = np.sum(fod_disrtibution, axis=-1)
    fod_disrtibution = np.divide(fod_disrtibution, amp_sums[..., np.newaxis], where=amp_sums[..., np.newaxis]!=0)

    parameters = {
        'Dw': dw,          # maximum diffusion coefficient
        'rho': rho,            # Proliferation rate
        'gm' : gm,
        'wm' : wm,    # Proliferation rate
        'diffusionTensors': fod_disrtibution, # diffusion tissue map as shown above
        'diffusionTensorExponent': 1, # exponent for the diffusion tensor, 1.0 for linear relationship
        'diffusionEllipsoidScaling':1,#21.713178343886213,
        'NxT1_pct': x,    # tumor position [%]
        'NyT1_pct': y,
        'NzT1_pct': z,
        'init_scale': init_scale, #scale of the initial gaussian
        'resolution_factor': resolution_factor, #resultion scaling for calculations
        'verbose': True, #printing timesteps 
        'time_series_solution_Nt': 64, # number of timesteps in the output
        'stopping_volume': stoppingVolume,
        'stopping_time': 2000, # 1000
        'use_homogen_gm': True,
        'RatioDw_Dg': RatioDw_Dg
    }

    start_time = time.time()
    fK_FOD_Solver = FK_FOD_Solver(parameters)
    result = fK_FOD_Solver.solve(doPlot=False)
    end_time = time.time()  # Store the end time
    execution_time = int(end_time - start_time)  # Calculate the difference

    nib.save(nib.Nifti1Image(result['final_state'].astype(np.float32), affine), f'./dataset/FK_FOD_result_{RatioDw_Dg}.nii.gz')
    return result


if __name__ == '__main__':
    X = 0.45
    Y = 0.3
    Z = 0.5
    """x = 0.6
        y = 0.3
        z = 0.50"""
    # x = 0.47
    # y = 0.30
    # z = 0.7
    dw = 1
    rho = 0.2#2
    Ratios_Dw_Dg = [10, 50, 100]

    tissueSegmentationPath = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/tissue_segments.nii.gz'

    originalTissue = nib.load(tissueSegmentationPath).get_fdata()
    affine = nib.load(tissueSegmentationPath).affine

    CSFMask = originalTissue == 1 # binary_dilation(originalTissue == 1, iterations = 1)

    tissue = originalTissue.copy()
    tissue[CSFMask] = 0

    gm = tissue == 2
    wm = tissue == 3
    gm[CSFMask] = 0
    wm[CSFMask] = 0

    init_scale = 0.1
    resolution_factor = 1# 0.6#1
    stoppingVolume =  15000

    for RatioDw_Dg in Ratios_Dw_Dg:
        print('%%%%%%%\nProcessing ratio', RatioDw_Dg)
        
        result_FOD = get_FK_FOD_solution(X, Y, Z, RatioDw_Dg, affine, dw, rho, gm, wm, init_scale, resolution_factor, stoppingVolume)
    
        resultDTI = get_FK_DTI_solution(X, Y, Z, RatioDw_Dg, affine, dw, rho, gm, wm, init_scale, resolution_factor, stoppingVolume)

        resultFK = get_FK_solver_solution(X, Y, Z, RatioDw_Dg, affine, dw, rho, gm, wm, init_scale, resolution_factor, stoppingVolume)

        x = int(tissue.shape[0]*X)
        y = int(tissue.shape[1]*Y)
        z = int(tissue.shape[2]*Z)


        fig, ax = plt.subplots(2,3, figsize=(12,6))
        ax[0,0].imshow(tissue[:,:,z]>0,alpha=0.5*(tissue[:,:,z]==0), cmap='gray')
        ax[0,0].imshow(result_FOD['final_state'][:,:,z], alpha=0.5*(result_FOD['final_state'][:,:,z]>0.0001), cmap = "Reds")	
        ax[1,0].imshow(tissue[:,y,:]>0,alpha=0.5*(tissue[:,y,:]==0), cmap='gray')
        ax[1,0].imshow(result_FOD['final_state'][:,y,:], alpha=0.5*(result_FOD['final_state'][:,y,:]>0.0001), cmap = "Reds")	

        # ax[0].colorbar()
        ax[0,0].set_title('Tumor FOD')

        ax[0,1].imshow(tissue[:,:,z]>0,alpha=0.5*(tissue[:,:,z]==0), cmap='gray')
        ax[0,1].imshow(resultDTI['final_state'][:,:,z], alpha=0.5*(resultFK['final_state'][:,:,z]>0.0001), cmap = "Reds")	
        ax[1,1].imshow(tissue[:,y,:]>0,alpha=0.5*(tissue[:,y,:]==0), cmap='gray')
        ax[1,1].imshow(resultDTI['final_state'][:,y,:], alpha=0.5*(resultFK['final_state'][:,y,:]>0.0001), cmap = "Reds")	
        ax[0,1].set_title('Tumor DTI')

        ax[0,2].imshow(tissue[:,:,z]>0,alpha=0.5*(tissue[:,:,z]==0), cmap='gray')
        ax[0,2].imshow(resultFK['final_state'][:,:,z], alpha=0.5*(resultFK['final_state'][:,:,z]>0.0001), cmap = "Reds")	
        ax[1,2].imshow(tissue[:,y,:]>0,alpha=0.5*(tissue[:,y,:]==0), cmap='gray')
        ax[1,2].imshow(resultFK['final_state'][:,y,:], alpha=0.5*(resultFK['final_state'][:,y,:]>0.0001), cmap = "Reds")	
        ax[0,2].set_title('Tumor FK')

        # fig.colorbar()
        # plt.show()

        fig.savefig(f'./dataset/fk_fod_vs_dti_vs_fk_ratioDw_Dg_{RatioDw_Dg}.png')
        # print('here')
