## The following is for running on JLSE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dh_posei
# mkdir one_peak_test
python deepHyp.py
#--peak "one_peak"  --n_points 450000 --json_file "One_peak.json"  --output_model "one_peak_test/"  --input_flag 1  --output_flag 1 --factor_reset 10 --n_iterations  20
conda deactivate
