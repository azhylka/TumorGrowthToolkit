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


for_each * : dwi2response msmt_5tt IN/T1w/Diffusion/data.nii.gz IN/T1w/Diffusion/5tt.nii.gz \
        IN/T1w/Diffusion/wm.txt IN/T1w/Diffusion/gm.txt IN/T1w/Diffusion/csf.txt \
        -fslgrad IN/T1w/Diffusion/bvecs IN/T1w/Diffusion/bvals -lmax 8,8,8,8 -nthreads 7 -force
responsemean */T1w/Diffusion/wm.txt group_average_wm_response.txt -force
