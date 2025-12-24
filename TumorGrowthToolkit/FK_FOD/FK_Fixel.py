import numpy as np
import os
import os.path as osp
import copy
from scipy.ndimage import zoom
from ..FK.FK import Solver as FK_Solver
import scipy.ndimage
from scipy.ndimage import binary_dilation
import nibabel as nib
import matplotlib.pyplot as plt
import itertools
from .utils import get_direction_to_index
import pickle
import time
import matplotlib.pyplot as plt

'''
Forward solver Fixel 

Andrey Zhylka - Adapting the DTI solver written by Jonas Weidner that was based on Michal Balcerak's solver
'''

class FK_Fixel_Solver(FK_Solver):
    def __init__(self, params):
        super().__init__(params)

    def filter_gm_csf(self, D_domain, wm, gm, exponent = 1 , linear = 0, ratioDw_Dg = None, desiredSTD = None):

        upper_limit = self.params.get('relative_upper_limit_DTI', 2)
        lower_limit = self.params.get('relative_lower_limit_DTI', 0)

        brainMask = np.logical_or(wm, gm)
        normalizationMask = wm > 0
        
        # if desiredSTD is not None:
        all_values = np.stack(list(D_domain.values()), axis=-1)[normalizationMask]
        all_mean = np.mean(all_values)
        all_std = np.std(all_values)


        for D_key, output in D_domain.items():
            # output = np.copy(D_flow)

            if desiredSTD is not None:
                mean = np.mean(output[normalizationMask])
                std = np.std(output[normalizationMask])
                # output[brainMask] = ((output[brainMask] - all_mean) / all_std) * desiredSTD + 1
                output[brainMask] = ((output[brainMask] - mean) / std) * desiredSTD + 1
            else:
                output[brainMask] /= np.mean(output[normalizationMask])

            if not (wm is None or gm  is None or ratioDw_Dg is None):
            
                print('set gm to uniform and wm to DTI')
                csfMask = np.logical_and(wm <= 0, gm <= 0)
                output[csfMask] = 0 
                gmThreshold = 1.0 / ratioDw_Dg  
                output[gm > 0 ] = gmThreshold # fix gray matter
                # borderMask = binary_dilation(csfMask, iterations = 1)
                # output[borderMask] = 0
                #clip wm to lowest gm
                output[np.logical_and(wm > 0, output < gmThreshold)] = gmThreshold

            # output[output<0] = 0
            # FIXME does it make sense to use exponent?
            # output = output ** exponent + linear * output

            output = np.clip(output, lower_limit, upper_limit)
            # output[np.logical_and( 
            #             np.repeat((brainMask > 0)[..., np.newaxis], repeats=output.shape[-1], axis=-1),
            #             output < lower_limit)] = lower_limit
            D_domain[D_key] = output

        return D_domain

    # def m_Tildas(self, rgbImg):
        
    #     retTildas = np.zeros_like(rgbImg)

    #     for i in range(3):
    #         retTildas[:,:,:,i] = (np.roll(rgbImg[:,:,:,i],-1,axis=i) + rgbImg[:,:,:,i])/2
        
    #     return retTildas

    def get_D_from_Fixel_DTI_style(self, fixel_dir, cache_dir, num_fixels=3):
        D = {}


        os.makedirs(cache_dir, exist_ok=True)
        cached_D = osp.join(cache_dir, 'Fixel_D_domain.pkl')

        if not osp.exists(cached_D):
            for fixel in range(num_fixels):
                fixels = nib.load(osp.join(fixel_dir, f'fixel_{fixel}.nii.gz')).get_fdata()
                fixel_amps = nib.load(osp.join(fixel_dir, f'amp_{fixel}.nii.gz')).get_fdata()
                fixels = fixels * fixel_amps[..., np.newaxis]

                closest_neighbors = nib.load(osp.join(fixel_dir, f'closest_fixel_{fixel}_to_fixels.nii.gz')).get_fdata()
                closest_neighbor_amps = nib.load(osp.join(fixel_dir, f'closest_fixel_{fixel}_to_fixels_amps.nii.gz')).get_fdata()
                closest_neighbors = closest_neighbors * closest_neighbor_amps[..., np.newaxis]

                for axis in range(3):
                    for offset_idx, offset in enumerate([-1, 1]):
                        neighbors = closest_neighbors[..., axis*2+offset_idx,:]
                        D_key = f'D{fixel}_{["minus","plus"][offset_idx==1]}_{["x","y","z"][axis]}'
                        D[D_key] = (np.abs(fixels[..., axis]) + np.abs(neighbors[..., axis])) / 2 # this way it is actually the flow through the face between voxels
            with open(cached_D, 'wb') as f:
                pickle.dump(D, f)
        else:
            with open(cached_D, 'rb') as f:
                D = pickle.load(f)
        return D
    
    
    def FK_update_DTI_style(self, A, Dpx, Dmx, Dpy, Dmy, Dpz, Dmz,
                            f, dt, cache_dir=None): #dx, dy, dz, fixel_ratios, cache_dir=None):
        # Optimized NumPy fallback (pre-computed rolls)
        SP = 0
        A_rolled = {
            'plus_x': np.roll(A, 1, axis=0),
            'minus_x': np.roll(A, -1, axis=0),
            'plus_y': np.roll(A, 1, axis=1),
            'minus_y': np.roll(A, -1, axis=1),
            'plus_z': np.roll(A, 1, axis=2),
            'minus_z': np.roll(A, -1, axis=2),
        }
        
        # Pre-compute differences (vectorized)
        diffs = {
            'x': A_rolled['plus_x'] - A,
            'minus_x': A - A_rolled['minus_x'],
            'y': A_rolled['plus_y'] - A,
            'minus_y': A - A_rolled['minus_y'],
            'z': A_rolled['plus_z'] - A,
            'minus_z': A - A_rolled['minus_z'],
        }

        # SP_x = 1/(dx*dx) * (Dpx* diffs['x'][...,np.newaxis] - Dmx* diffs['minus_x'][...,np.newaxis]) 
        # SP_y = 1/(dy*dy) * (Dpy* diffs['y'][...,np.newaxis] - Dmy* diffs['minus_y'][...,np.newaxis])
        # SP_z = 1/(dz*dz) * (Dpz* diffs['z'][...,np.newaxis] - Dmz* diffs['minus_z'][...,np.newaxis])

        SP = (Dpx* diffs['x'][...,np.newaxis] - Dmx* diffs['minus_x'][...,np.newaxis]
            + Dpy* diffs['y'][...,np.newaxis] - Dmy* diffs['minus_y'][...,np.newaxis]
            + Dpz* diffs['z'][...,np.newaxis] - Dmz* diffs['minus_z'][...,np.newaxis]).sum(axis=-1)

        # SP = np.sum(SP_x + SP_y + SP_z, axis=-1)
        # SP = np.sum(fixel_ratios * (SP_x + SP_y + SP_z), axis=-1)
        
        diff_A = (SP + f*np.multiply(A,1-A)) * dt
        A += diff_A
        return A

    def crop_tissues_and_tumor(self, tissue, tumor_initial, brainmask,  margin=2, threshold=0.0):
        """
        Crop Tissue and tumor_initial such that we remove the maximal amount of voxels
        where the tissue is lower than0 the threshold.
        A margin is left around the tissues.

        :param Tissue: 4D numpy array of diffusion direction (RGB file)
        :param tumor_initial: 3D numpy array of initial tumor
        :param margin: Margin to leave around the tissues
        :param threshold: Threshold to consider as no tissue
        :return: Cropped tissue, tumor_initial, and the crop coordinates
        """

        # Finding indices where the tissue sum is greater than to the threshold
        tissue_indices = np.argwhere(brainmask > threshold)

        # Finding the bounding box for cropping, considering the margin
        min_coords = np.maximum(tissue_indices.min(axis=0) - margin, 0)
        max_coords = np.minimum(tissue_indices.max(axis=0) + margin + 1, brainmask.shape)

        # Cropping tissue and tumor_initial
        cropped_tissue = {}
        for k, v in tissue.items():
            cropped_tissue[k] = v[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    
        cropped_tumor_initial = tumor_initial[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

        return cropped_tissue, cropped_tumor_initial, (min_coords, max_coords)

    def solve(self, doPlot = False):

        if doPlot:
            self.doPlot = True
        else:
            self.doPlot = False

        # Unpack parameters
        stopping_time = self.params.get('stopping_time', 100)
        stopping_volume = self.params.get('stopping_volume', np.inf) #mm^3

        Dw = self.params['Dw']
        f = self.params['rho']

        D_domain = self.get_D_from_Fixel_DTI_style(self.params['fixel_dir'], self.params.get('cache_dir', None))
        print('D_domain shape:', D_domain['D0_minus_x'].shape)

        # diffusionEllipsoidScaling = self.params.get('diffusionEllipsoidScaling', 1.)
        # print(f'diffusionEllipsoidScaling: {diffusionEllipsoidScaling}')
        desiredSTD = self.params.get('desiredSTD', None)

        diffusionTensorExponent = self.params.get('diffusionTensorExponent', 1) # 3 was a good value but 1 is plain linear
        diffusionTensorLinear = self.params.get('diffusionTensorLinear', 0)

        days = self.params.get('days', 100) # Normalized time at 100d, should stay at 100 otherwise it is overparameterized

        NxT1_pct = self.params['NxT1_pct']
        NyT1_pct = self.params['NyT1_pct']
        NzT1_pct = self.params['NzT1_pct']
        res_factor = self.params['resolution_factor']  #Res scaling
        # th_matter = self.params.get('th_matter', 0.1) # not yet used
        dx_mm = self.params.get('dx_mm', 1.)  #default 1mm
        dy_mm = self.params.get('dy_mm', 1.)  
        dz_mm = self.params.get('dz_mm', 1.)
        init_scale  = self.params.get('init_scale', 1.)
        time_series_solution_Nt = self.params.get('time_series_solution_Nt', None) #record timeseries, number of steps
        verbose = self.params.get('verbose', False)  

        # print("debug start transform")
        # # Apply the transformation
        # if diffusionEllipsoidScaling == 1:
        #     tensor_array_prime = self.params["diffusionTensors"]
        # else:
        #     tensor_array_prime = tools.elongate_tensor_along_main_axis_torch(self.params["diffusionTensors"], diffusionEllipsoidScaling)

        # #print("debug end transform")

        if self.params.get('use_homogen_gm', False):
            sGM = zoom(self.params['gm'], res_factor, order=0)
            sWM = zoom(self.params['wm'], res_factor, order=0)
            print('WM mask shape', sWM.shape, 'after applying res_factor', res_factor, 
                  'before applying', self.params['wm'].shape)

            ratioDw_Dg = self.params.get('RatioDw_Dg', 10.)

            # TODO fix tools...
            D_domain = self.filter_gm_csf(D_domain, wm = sWM, gm = sGM, exponent = diffusionTensorExponent , linear = diffusionTensorLinear, 
                                      ratioDw_Dg = ratioDw_Dg, desiredSTD = desiredSTD)
        else:
            # TODO fix tools...
            D_domain = self.filter_gm_csf(D_domain, exponent = diffusionTensorExponent , linear = diffusionTensorLinear) #, desiredSTD = desiredSTD)

        # Validate input
        assert isinstance(D_domain, dict), "sRGB must be a numpy array"
        assert all(map(lambda x: x.ndim == 3, D_domain.values()))
        assert 0 <= NxT1_pct <= 1, "NxT1_pct must be between 0 and 1"
        assert 0 <= NyT1_pct <= 1, "NyT1_pct must be between 0 and 1"
        assert 0 <= NzT1_pct <= 1, "NzT1_pct must be between 0 and 1"

        brainmask = np.logical_or(self.params['gm'], self.params['wm'])
        brainmask_low_res = np.logical_or(sGM, sWM)
        original_shape = brainmask.shape
        new_shape = brainmask_low_res.shape

        # Calculate the zoom factor for each dimension
        extrapolate_factor = tuple(float(orig_sz) / new_sz for new_sz, orig_sz in zip(new_shape, original_shape))

        # Update grid size and steps for low resolution
        # use one of the expected keywords 
        Nx, Ny, Nz = D_domain['D0_minus_x'].shape  #low_res_tissue_constrained_D.shape[:3]
        print('Grid size after res_factor', res_factor, ':', Nx, Ny, Nz)

        # Adjust grid steps based on zoom factor
        dx = dx_mm / res_factor
        dy = dy_mm / res_factor
        dz = dz_mm / res_factor

        # Calculate the absolute positions based on percentages
        NxT1 = int(NxT1_pct * Nx)
        NyT1 = int(NyT1_pct * Ny)
        NzT1 = int(NzT1_pct * Nz)

        #stability condition \Delta t \leq \min \left( \frac{\Delta x^2}{6 D_{\text{max}}}, \frac{1}{\rho} \right)
        D_max = np.max(list(map(lambda x: x.max(), D_domain.values())))
        Nt = np.max([stopping_time * Dw * D_max/np.power((np.min([dx,dy,dz])),2)*8 + 100, stopping_time * f *1.1 ]) 
        dt = stopping_time/Nt
        N_simulation_steps = int(np.ceil(Nt))
        if verbose: 
            print(f'Number of simulation timesteps: {N_simulation_steps}')

        xv, yv, zv = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing='ij')
        A = np.array(self.gauss_sol3d(xv - NxT1, yv - NyT1, zv - NzT1,dx,dy,dz,init_scale))
        print("init: ",A.shape, "Volume of init Tumor", np.sum(A))
        col_res = np.zeros([2, Nx, Ny, Nz])
        col_res[0] = copy.deepcopy(A) #init
        
        #cropping
        D_domain_cropped, A, (min_coords, max_coords) = self.crop_tissues_and_tumor(D_domain, A, brainmask_low_res,
                                                                                     margin=2, threshold=0.5)
        # Simulation code
        result = {}
        fixel_ratios = nib.load(osp.join(self.params['fixel_dir'], 'fixel_ratio_volume.nii.gz')).get_fdata()
        cropped_fixel_ratios = fixel_ratios[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
        
        # Initialize time series list if needed
        time_series_data = [] if time_series_solution_Nt is not None else None

        # Determine the steps at which to record the data
        if time_series_data is not None:
            # Using linspace to get exact steps to record, including first and last
            record_steps = np.linspace(0, N_simulation_steps - 1, time_series_solution_Nt, dtype=int)

        #print("debug start simulation")
        try:
            finalTime = None
            result['success'] = False
            
            #check if origin within brainmask
            if not brainmask_low_res[NxT1, NyT1, NzT1]:
                fig = plt.figure()
                plt.imshow(brainmask_low_res[:,:,NzT1], cmap='gray')
                plt.scatter(NxT1, NyT1, NzT1, 'r')
                plt.savefig('/mnt/Drive2/andrey/brainmask_origin_error.png')
                raise ValueError("Origin not within brainmask", NxT1, NyT1, NzT1, '\nratios are', NxT1_pct, NyT1_pct, NzT1_pct,
                                'the shape is', brainmask_low_res.shape)
            
            print('Starting simulation for', N_simulation_steps, 'steps')

            Dpx = np.stack([D_domain_cropped["D0_plus_x"], D_domain_cropped["D1_plus_x"], D_domain_cropped["D2_plus_x"]], axis=-1)
            Dpx *= cropped_fixel_ratios
            nib.save(nib.Nifti1Image(Dpx[...,0].astype(np.float32), np.eye(4)), 
                     osp.join(self.params['cache_dir'], f'SP_Fixel_std{self.params["desiredSTD"]}.nii.gz'))

            Dpx /= (dx * dx)
            Dmx = np.stack([D_domain_cropped["D0_minus_x"], D_domain_cropped["D1_minus_x"], D_domain_cropped["D2_minus_x"]], axis=-1)
            Dmx *= cropped_fixel_ratios
            Dmx /= (dx * dx)

            Dpy = np.stack([D_domain_cropped["D0_plus_y"], D_domain_cropped["D1_plus_y"], D_domain_cropped["D2_plus_y"]], axis=-1)
            Dpy *= cropped_fixel_ratios
            Dpy /= (dy * dy)
            Dmy = np.stack([D_domain_cropped["D0_minus_y"], D_domain_cropped["D1_minus_y"], D_domain_cropped["D2_minus_y"]], axis=-1)
            Dmy *= cropped_fixel_ratios
            Dmy /= (dy * dy)

            Dpz = np.stack([D_domain_cropped["D0_plus_z"], D_domain_cropped["D1_plus_z"], D_domain_cropped["D2_plus_z"]], axis=-1)
            Dpz *= cropped_fixel_ratios
            Dpz /= (dz * dz)
            Dmz = np.stack([D_domain_cropped["D0_minus_z"], D_domain_cropped["D1_minus_z"], D_domain_cropped["D2_minus_z"]], axis=-1)
            Dmz *= cropped_fixel_ratios
            Dmz /= (dz * dz)
            
            for t in range(N_simulation_steps):
                A_Old_size = np.sum(A)
                # oldA = copy.deepcopy(A)
                start_time = time.time()
                A = self.FK_update_DTI_style(A, Dpx, Dmx, Dpy, Dmy, Dpz, Dmz,
                                            f, dt, cache_dir=self.params['cache_dir']) # dx, dy, dz, cropped_fixel_ratios, 
                # if t > 30:
                #     plt.imsave(osp.join(self.params['cache_dir'], f'tumor_step_{t}_fixel.png'), A[:,:,NzT1], cmap='hot')
                
                end_time = time.time()
                if t % 100 == 0:
                    print('Time for update step', t, ':', end_time - start_time, 'seconds.')

                # if t % 100 == 0:
                #     plt.imshow((A-oldA)[:,:,NzT1])
                #     plt.title(f"Update at step {t}")
                #     plt.show()
                A = np.abs(A)
                volume = dx * dy * dz * np.sum(A)
                if volume >= stopping_volume:
                    finalTime = t * dt
                    break

                diffA = np.sum(A) - A_Old_size
                if  diffA < -10:
                    print("Tumor is shrinking at time", t*dt, "by", diffA)
                    result['success'] = False
                    break

                if volume < 0.000001:
                    print("Volume is to small")
                    result['success'] = False
                    break

                # if verbose and t % 1000 == 0:
                #     imshow_slice = cropped_FOD[:,:,int(NzT1_pct * A.shape[2]), 0]
                #     imshow_slice /= np.max(imshow_slice)
                #     from matplotlib import pyplot as plt   

                #     plt.imshow(imshow_slice)
                #     plt.imshow(A[:,:,int(NzT1_pct * A.shape[2])], alpha=0.5*(A[:,:,int(NzT1_pct * A.shape[2])]>0.001), cmap='hot', vmin=0, vmax=1)
                #     plt.show()
                #     if diffA < 0:
                #         print("Tumor is shrinking at time", t*dt, "by", diffA)
                #         #plt.imshow(imshow_slice)
                #         diffAIMG = np.abs(A - oldA)
                #         comz = scipy.ndimage.measurements.center_of_mass(diffAIMG)[2]
                #         plt.imshow(diffAIMG[:,:,int(comz)], alpha=0.5*(diffAIMG[:,:,int(comz)]>0.001), cmap='hot')
                #         plt.title("Diff")
                #         plt.show()

                # Record data at specified steps
                if time_series_data is not None:
                    if t in record_steps:
                        time_series_data.append(copy.deepcopy(A))
            
            if finalTime is None:
                finalTime = stopping_time
            
            # Process final state
            A = self.restore_tumor((Nx, Ny, Nz), A, (min_coords, max_coords))
            col_res[1] = copy.deepcopy(A)  # final

            # Save results in the result dictionary
            result['initial_state'] = np.array(zoom(col_res[0], extrapolate_factor, order=1))
            result['final_state'] = np.array(zoom(col_res[1], extrapolate_factor, order=1))
            result['final_time'] = finalTime
            result['final_volume'] = volume
            result['stopping_criteria'] = 'volume' if volume >= stopping_volume else 'time'
            result['time_series'] = np.array([zoom(self.restore_tumor((Nx, Ny, Nz), state, (min_coords, max_coords)), extrapolate_factor, order=1)
                                            for state in time_series_data]) if time_series_data is not None else None
            result['Dw'] = Dw
            result['rho'] = f
            result['success'] = True
                
                    
        except Exception as e:
            print(e)
            result['error'] = str(e)
            result['success'] = False

        return result


# %%
