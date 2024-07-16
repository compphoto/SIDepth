#!/bin/bash
#SBATCH --job-name=CVD     	    # Name that will show up in squeue
#SBATCH --gres=gpu:1                   # Request 4 GPU "generic resource"
#SBATCH --time=7-00:00       		    # Max job time is 3 hours
#SBATCH --output=./logs/%N-%j.out   	# Terminal output to file named (hostname)-(jobid).out
#SBATCH --error=./logs/%N-%j.err		# Error log with format (hostname)-(jobid).out
#SBATCH --partition=long     		    # Partitions (long -- 7 days low priority, short -- 2 days intermediate priority, debug -- 4 hours high priority runtime)
#SBATCH --mail-type=all       		    # Send email on job start, end and fail
#SBATCH --mail-user=mahesh_reddy@sfu.ca
#SBATCH --qos=overcap
#SBATCH --nodelist=cs-venus-01

# The SBATCH directives above set options similarly to command line arguments to srun
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate detectron
echo "Environment activated"

echo "Loading colmap..."
module load TOOLS/COLMAP/20210324-CUDA10

# python boosting.py --data_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/crops/bicycle/ --output_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/depth/bicycle --depthNet 0 --Final --colorize_results
python boosting.py --data_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/crops/bicycle/ --output_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/depth_r20/bicycle --depthNet 0 --R20 --colorize_results

# python boosting.py --data_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/crops/motorcycle/ --output_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/depth/motorcycle --depthNet 0 --Final --colorize_results
# python boosting.py --data_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/crops/motorcycle/ --output_dir /project/aksoy-lab/datasets/RGBD/unsplash/images/depth_r20/motorcycle --depthNet 0 --R20 --colorize_results
