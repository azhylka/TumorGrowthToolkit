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

X = 0.45
Y = 0.3
Z = 0.5
Ratios_Dw_Dg = [10, 50, 100]


for RatioDw_Dg in Ratios_Dw_Dg:
    print('%%%%%%%\nProcessing ratio', RatioDw_Dg)
    # tissueSegmentationPath = "./dataset/sub-mni152_tissue-with-antsN4_space-sri.nii.gz"
    # tensorPath = "./dataset/HCP1065_FOD_sphere362_segmented_amps.nii.gz"
    # fod_26dir_file = './dataset/hcp_regrid_segmented_amps.nii.gz'

    tissueSegmentationPath = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/tissue_segments.nii.gz'
    # tensorPath = "./dataset/HCP1065_FOD_sphere362_segmented_amps.nii.gz"
    fod_26dir_file = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/wmfod_norm_sphere362_segmented_amps.nii.gz'
    # './dataset/hcp_regrid_segmented_amps.nii.gz'

    originalTissue = nib.load(tissueSegmentationPath).get_fdata()
    affine = nib.load(tissueSegmentationPath).affine

    #create a 3x3 tensor for each voxel
    # tissueTensor = tools.get_tensor_from_lower6(nib.load(tensorPath).get_fdata()[:,:,:,0,:])
    fod_disrtibution = nib.load(fod_26dir_file).get_fdata()

    amp_sums = np.sum(fod_disrtibution, axis=-1)
    fod_disrtibution = np.divide(fod_disrtibution, amp_sums[..., np.newaxis], where=amp_sums[..., np.newaxis]!=0)

    CSFMask = originalTissue == 1 # binary_dilation(originalTissue == 1, iterations = 1)

    tissue = originalTissue.copy()
    tissue[CSFMask] = 0
    # tissueTensor[CSFMask] = 0

    dw = 1
    rho = 0.2#2
    #ventricle
    """x = 0.6
    y = 0.3
    z = 0.50"""
    # x = 0.47
    # y = 0.30
    # z = 0.7
    x = X
    y = Y
    z = Z

    gm = tissue == 2
    wm = tissue == 3
    gm[CSFMask] = 0
    wm[CSFMask] = 0

    init_scale = 0.1
    resolution_factor = 1# 0.6#1
    stoppingVolume =  15000
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

    x = int(tissue.shape[0]*parameters["NxT1_pct"])
    y = int(tissue.shape[1]*parameters["NyT1_pct"])
    z = int(tissue.shape[2]*parameters["NzT1_pct"])


    start_time = time.time()
    fK_FOD_Solver = FK_FOD_Solver(parameters)
    result = fK_FOD_Solver.solve(doPlot=False)
    end_time = time.time()  # Store the end time
    execution_time = int(end_time - start_time)  # Calculate the difference

    nib.save(nib.Nifti1Image(result['final_state'].astype(np.float32), affine), f'./dataset/FK_FOD_result_{RatioDw_Dg}.nii.gz')

    # %%%%%
    # tensorPath = "./dataset/FSL_HCP1065_tensor_1mm_space-HPC-AntsIndexSpace_SRI.nii.gz"

    # #create a 3x3 tensor for each voxel
    # tissueTensor = tools.get_tensor_from_lower6(nib.load(tensorPath).get_fdata()[:,:,:,0,:])
    tensorPath = '/Users/azhylka/Projects/TUMor_Data/HCP/100307/dt.nii.gz'
    tissueTensor = tools.get_tensor_from_lower6_mrtrix(nib.load(tensorPath).get_fdata()[:,:,:,:])
    tissueTensor[np.isnan(tissueTensor)] = 0


    # x = 0.47
    # y = 0.30
    # z = 0.7
    x = X
    y = Y
    z = Z

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

    # parametersFK = {
    #     'Dw': dw,          # maximum diffusion coefficient
    #     'rho': rho,        # Proliferation rate
    #     'gm' : gm,
    #     'wm' : wm,
    #     'NxT1_pct': parameters["NxT1_pct"],    # tumor position [%]
    #     'NyT1_pct': parameters["NyT1_pct"],
    #     'NzT1_pct': parameters["NzT1_pct"],
    #     'init_scale': init_scale, #scale of the initial gaussian
    #     'resolution_factor': resolution_factor, #resultion scaling for calculations
    #     'verbose': True, #printing timesteps 
    #     'time_series_solution_Nt': 64, # number of timesteps in the output
    #     'stopping_volume': stoppingVolume,
    #     'stopping_time': 1000,
    #     'use_homogen_gm': True,
    # }

    x = int(tissue.shape[0]*parameters["NxT1_pct"])
    y = int(tissue.shape[1]*parameters["NyT1_pct"])
    z = int(tissue.shape[2]*parameters["NzT1_pct"])

    dtiSolver = FK_DTI_Solver(parameters)
    resultDTI = dtiSolver.solve()
    nib.save(nib.Nifti1Image(resultDTI['final_state'].astype(np.float32), affine), f'./dataset/FK_DTI_result_{RatioDw_Dg}.nii.gz')

    # x = 0.47
    # y = 0.30
    # z = 0.7
    x = X
    y = Y
    z = Z

    parametersFK = {
        'Dw': dw,          # maximum diffusion coefficient
        'rho': rho,        # Proliferation rate
        'gm' : gm,
        'wm' : wm,
        'NxT1_pct': parameters["NxT1_pct"],    # tumor position [%]
        'NyT1_pct': parameters["NyT1_pct"],
        'NzT1_pct': parameters["NzT1_pct"],
        'init_scale': init_scale, #scale of the initial gaussian
        'resolution_factor': resolution_factor, #resultion scaling for calculations
        'verbose': True, #printing timesteps 
        'time_series_solution_Nt': 64, # number of timesteps in the output
        'stopping_volume': stoppingVolume,
        'stopping_time': 1000,
        'use_homogen_gm': True,
        'RatioDw_Dg': RatioDw_Dg
    }

    x = int(tissue.shape[0]*parameters["NxT1_pct"])
    y = int(tissue.shape[1]*parameters["NyT1_pct"])
    z = int(tissue.shape[2]*parameters["NzT1_pct"])

    fkSolver = FK_Solver(parametersFK)
    resultFK = fkSolver.solve()
    nib.save(nib.Nifti1Image(resultFK['final_state'].astype(np.float32), affine), f'./dataset/FK_result_{RatioDw_Dg}.nii.gz')


    fig, ax = plt.subplots(2,3, figsize=(12,6))
    ax[0,0].imshow(tissue[:,:,z]>0,alpha=0.5*(tissue[:,:,z]==0), cmap='gray')
    ax[0,0].imshow(result['final_state'][:,:,z], alpha=0.5*(result['final_state'][:,:,z]>0.0001), cmap = "Reds")	
    ax[1,0].imshow(tissue[:,y,:]>0,alpha=0.5*(tissue[:,y,:]==0), cmap='gray')
    ax[1,0].imshow(result['final_state'][:,y,:], alpha=0.5*(result['final_state'][:,y,:]>0.0001), cmap = "Reds")	

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
