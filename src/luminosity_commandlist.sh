#!/bin/sh

# Run the experiment in Slurm cluster:
# module load julia
# sh box_luminosity_commandlist.sh > cmds.txt
# sbatch_commandlist -t 8:00:00 -mem 4000 -A project_2001274 -commands cmds.txt

CMD="julia"

iterations=500000
output_base="out/luminosity500k"

for res in Stratified StratifiedPartition \
Systematic SystematicPartition \
SSP SSPPartition \
Killing Multinomial Residual; do
    for N in 8 16 32 64 128 256; do
          echo $CMD run_luminosity_experiment.jl -N $N --resampling $res \
 --iterations $iterations --output ${output_base}_${res}_$N.jld2
    done
done
