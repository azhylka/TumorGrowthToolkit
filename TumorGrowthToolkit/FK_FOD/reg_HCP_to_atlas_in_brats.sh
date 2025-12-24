#!/bin/bash

regged_old_atlas_dir=/home/brats_good_registerd_atlas/
out_folder=/home/HCPTemplate2BraTS
template_data_dir=/home/HCP_case/
code_dir=/home/code
python ${code_dir}/combine_5tt.py combine ${template_data_dir}/5tt_1.25.nii.gz ${template_data_dir}/5tt_combined.nii.gz

cd $regged_old_atlas_dir

for subj_dir in BraTS*/
# cd $out_folder
# for subj_dir in */
do
subj_id="${subj_dir%/}"

subj_out=${out_folder}/${subj_id}
mkdir ${subj_out}


antsRegistrationSyN.sh -t r -d 3 -f ${regged_old_atlas_dir}/${subj_id}/transformed_t1.nii.gz -m ${template_data_dir}/T1w_masked.nii.gz -o ${subj_out}/hcp2atlas_in_brats_rig -n 20

ConvertTransformFile 3 ${subj_out}/hcp2atlas_in_brats_rig0GenericAffine.mat ${subj_out}/hcp2atlas_in_brats_rig.mat --hm
sed -n '1,3p' ${subj_out}/hcp2atlas_in_brats_rig.mat > ${subj_out}/hcp2atlas_in_brats_rig.txt

mrtransform ${template_data_dir}/data.mif ${subj_out}/data_rig.mif -linear ${subj_out}/hcp2atlas_in_brats_rig.txt -force

mrtransform ${template_data_dir}/T1w_masked.nii.gz ${subj_out}/T1w_masked_rig.nii.gz -linear ${subj_out}/hcp2atlas_in_brats_rig.txt -force

mrconvert ${subj_out}/data_rig.mif ${subj_out}/data_rig.nii.gz --export_grad_fsl ${subj_out}/data_rig.bvec ${subj_out}/data_rig.bval -force

rm ${subj_out}/data_rig.mif

mrtransform ${template_data_dir}/5tt_combined.nii.gz ${subj_out}/5tt_combined_rig.nii.gz -linear ${subj_out}/hcp2atlas_in_brats_rig.txt -force

/opt/fsl/bin/fslsplit ${subj_out}/data_rig.nii.gz ${subj_out}/data_rig_shell -t

FIXED=${regged_old_atlas_dir}/${subj_id}/transformed_t1.nii.gz
MOVING=${subj_out}/T1w_masked_rig.nii.gz

antsRegistration -d 3 \
	  -r [${FIXED}, ${MOVING}, 1] \
  -m MI[${FIXED}, ${MOVING}, 1, 32] \
	    -t Rigid[0.1] \
	      -c [1000x500x250x100,1] \
	        -s 3x2x1x0 \
		  -f 8x4x2x1 \
		  -m MI[${FIXED}, ${MOVING}, 1, 32] \
	      -t Affine[0.1] \
	        -c [1000x500x250x100,1] \
		  -s 3x2x1x0 \
		    -f 8x4x2x1 \
      -m CC[${FIXED}, ${MOVING}, 1, 2] \
	        -t SyN[0.1, 3, 0] \
		  -c [100x70x50x20,1] \
		    -s 3x2x1x0 \
		      -f 8x4x2x1 \
		      -o [${subj_out}/data_rig_SyN_shell0000, ${subj_out}/T1w_masked_rig_sri.nii.gz, ${subj_out}/transformed_t1_in_rig.nii.gz]


cd ${subj_out}
parallel -j 10 \
   	"antsApplyTransforms -i {} -d 3 -e 0 \
	-t data_rig_SyN_shell00001Warp.nii.gz \
   -t data_rig_SyN_shell00000GenericAffine.mat \
   -r T1w_masked_rig_sri.nii.gz \
   -o trans_{} --float -v" \
 ::: data_rig_shell*.nii.gz
cd -

/opt/fsl/bin/fslmerge -t ${subj_out}/trans_data_rig.nii.gz ${subj_out}/trans_data_rig_shell*.nii.gz

rm ${subj_out}/data_rig_shell*
rm ${subj_out}/trans_data_rig_shell*
rm ${subj_out}/data_rig.nii.gz

dwigradcheck ${subj_out}/trans_data_rig.nii.gz -fslgrad ${subj_out}/data_rig.bvec ${subj_out}/data_rig.bval -export_grad_mrtrix ${subj_out}/fixed_trans_data_rig.bvec -nthreads 20 -force

dwi2tensor ${subj_out}/trans_data_rig.nii.gz ${subj_out}/trans_data_rig_dt.nii.gz -grad ${subj_out}/fixed_trans_data_rig.bvec

antsApplyTransforms -i ${subj_out}/5tt_combined_rig.nii.gz -d 3 -e 0 \
   -t ${subj_out}/data_rig_SyN_shell00001Warp.nii.gz \
   -t ${subj_out}/data_rig_SyN_shell00000GenericAffine.mat \
   -r ${subj_out}/T1w_masked_rig_sri.nii.gz \
   -o ${subj_out}/5tt_combined_rig_SyN.nii.gz -v -n NearestNeighbor

python ${code_dir}/combine_5tt.py decompose ${subj_out}/5tt_combined_rig_SyN.nii.gz 5 ${subj_out}/5tt_rig_SyN.nii.gz

dwi2mask ${subj_out}/trans_data_rig.nii.gz -grad ${subj_out}/fixed_trans_data_rig.bvec ${subj_out}/trans_data_rig_mask.nii.gz -force

dwi2response msmt_5tt ${subj_out}/trans_data_rig.nii.gz ${subj_out}/5tt_rig_SyN.nii.gz ${subj_out}/response_wm.txt ${subj_out}/response_csf.txt ${subj_out}/response_gm.txt -grad ${subj_out}/fixed_trans_data_rig.bvec -nthreads 20 -info -force

dwi2fod msmt_csd ${subj_out}/trans_data_rig.nii.gz -grad ${subj_out}/fixed_trans_data_rig.bvec ${subj_out}/response_wm.txt ${subj_out}/wmfod.nii.gz ${subj_out}/response_gm.txt ${subj_out}/gm.nii.gz ${subj_out}/response_csf.txt ${subj_out}/csf.nii.gz -mask ${subj_out}/trans_data_rig_mask.nii.gz -nthreads 20 -force

fod2fixel  ${subj_out}/wmfod.nii.gz ${subj_out}/fixels -nii -peak_amp amp.nii.gz -disp disp.nii.gz -nthreads 20 -force

mrgrid ${subj_out}/trans_data_rig.nii.gz regrid -voxel 1.5 ${subj_out}/trans_data_rig_1.5.nii.gz
rm ${subj_out}/trans_data_rig.nii.gz

dwi2mask ${subj_out}/trans_data_rig_1.5.nii.gz -grad ${subj_out}/fixed_trans_data_rig.bvec ${subj_out}/trans_data_rig_mask_1.5.nii.gz -force
dwi2tensor ${subj_out}/trans_data_rig_1.5.nii.gz ${subj_out}/trans_data_rig_1.5_dt.nii.gz \
     -grad ${subj_out}/fixed_trans_data_rig.bvec -mask ${subj_out}/trans_data_rig_mask_1.5.nii.gz -force
dwi2response msmt_5tt ${subj_out}/trans_data_rig_1.5.nii.gz ${subj_out}/5tt_rig_SyN.nii.gz \
     ${subj_out}/response_wm_1.5.txt \
     ${subj_out}/response_csf_1.5.txt \
     ${subj_out}/response_gm_1.5.txt \
     -grad ${subj_out}/fixed_trans_data_rig.bvec -nthreads 20 -info -force
dwi2fod msmt_csd ${subj_out}/trans_data_rig_1.5.nii.gz -grad ${subj_out}/fixed_trans_data_rig.bvec \
     ${subj_out}/response_wm_1.5.txt ${subj_out}/wmfod_1.5.nii.gz \
     ${subj_out}/response_gm_1.5.txt ${subj_out}/gm_1.5.nii.gz \
     ${subj_out}/response_csf_1.5.txt ${subj_out}/csf_1.5.nii.gz \
     -mask ${subj_out}/trans_data_rig_mask_1.5.nii.gz -nthreads 20 -force

rm ${subj_out}/trans_data_rig_1.5.nii.gz

fod2fixel  ${subj_out}/wmfod_1.5.nii.gz ${subj_out}/fixels_15 -nii -peak_amp amp.nii.gz -disp disp.nii.gz -nthreads 20 -force

done

