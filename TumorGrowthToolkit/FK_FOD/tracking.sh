data_dir=/Volumes/anjssd/HCP_TemplateGroup/

cd $data_dir/Diffusion/

for subj_dir in */
do

	sid="${subj_dir%/}"

	cd $subj_dir
    echo $subj_dir
	
	5ttgen freesurfer $data_dir/Structural/$subj_dir/T1w/aparc.a2009s+aseg.nii.gz $data_dir/Diffusion/${sid}/T1w/Diffusion/5tt.nii.gz -lut $data_dir/FreeSurferColorLUT.txt -force -nthreads 5
	
    cd ..
done

cd $data_dir/Diffusion/

for_each * : dwi2mask IN/T1w/Diffusion/data.nii.gz IN/T1w/Diffusion/mask.nii.gz \
        -fslgrad IN/T1w/Diffusion/bvecs IN/T1w/Diffusion/bvals -force

for_each * : dwi2response msmt_5tt IN/T1w/Diffusion/data.nii.gz IN/T1w/Diffusion/5tt.nii.gz \
        IN/T1w/Diffusion/wm.txt IN/T1w/Diffusion/gm.txt IN/T1w/Diffusion/csf.txt \
        -fslgrad IN/T1w/Diffusion/bvecs IN/T1w/Diffusion/bvals -nthreads 7 -force

responsemean */T1w/Diffusion/wm.txt ../group_average_wm_response.txt -force
responsemean */T1w/Diffusion/gm.txt ../group_average_gm_response.txt -force
responsemean */T1w/Diffusion/csf.txt ../group_average_csf_response.txt -force

for_each * : dwi2fod msmt_csd IN/T1w/Diffusion/data.nii.gz \
        ../group_average_wm_response.txt IN/T1w/Diffusion/wmfod.mif \
        ../group_average_gm_response.txt IN/T1w/Diffusion/gm.mif  \
        ../group_average_csf_response.txt IN/T1w/Diffusion/csf.mif \
        -mask IN/T1w/Diffusion/nodif_brain_mask.nii.gz \
        -fslgrad IN/T1w/Diffusion/bvecs IN/T1w/Diffusion/bvals \
        -nthreads 7 -force

for_each * : mtnormalise IN/T1w/Diffusion/wmfod.mif IN/T1w/Diffusion/wmfod_norm.mif \
        IN/T1w/Diffusion/gm.mif IN/T1w/Diffusion/gm_norm.mif \
        IN/T1w/Diffusion/csf.mif IN/T1w/Diffusion/csf_norm.mif \
        -mask IN/T1w/Diffusion/mask.nii.gz -nthreads 7 -force

mkdir -p ../template/fod_input
mkdir ../template/mask_input

for_each * : ln -s ../../IN/T1w/Diffusion/wmfod_norm.mif "../template/fod_input/PRE.mif"
for_each * : ln -s ../../IN/T1w/Diffusion/mask.nii.gz "../template/mask_input/PRE.nii.gz"

population_template ../template/fod_input \
        -mask_dir ../template/mask_input ../template/wmfod_template.mif \
        -voxel_size 1.25 -nthreads 7 -force

