#script for ugent HPC
cd /kyukon/data/gent/courses/2025/mach_learn_C003758/members/vsc45329/nemotron-CoT-experiments
module load vsc-venv; source vsc-venv --activate --requirements ./hpc/requirements.txt --modules ./hpc/modules.txt
HF_HOME=./hf_home python train_model.py