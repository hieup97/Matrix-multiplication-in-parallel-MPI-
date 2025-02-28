all:	job

login:
	module load slurm/17.11.12
milan:
	module load slurm/seawulf3/21.08.8
job:
	sbatch job.sh
check:
	squeue -u ${USER}
show:
	scontrol show job
rm:
	rm -rf *.log *.out *.exe
