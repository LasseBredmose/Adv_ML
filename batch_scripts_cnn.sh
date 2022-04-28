#!/bin/sh 


python3 -m pip install --user -r requirements.txt
small_test=0
transformation=0
maxpool=0

# Load dependencies
module load python3/3.9.6
# module load matplotlib/3.4.2-numpy-1.21.1-python-3.9.6
module load pandas/1.3.1-python-3.9.6
module load cuda/11.5
module load cudnn/v8.3.0.98-prod-cuda-11.5
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set the job Name -- 
#BSUB -J 5_CNN_27_4_mp_0_tran_0
### -- ask for number of cores (default: 1) -- 
#BSUB -n 2 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=12GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o 5_CNN_27_4_mp_0_tran_0.out
#BSUB -e 5_CNN_27_4_mp_0_tran_0.err

# here follow the commands you want to execute 
python3 main.py train $small_test $transformation $maxpool

