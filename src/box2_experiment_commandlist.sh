#!/bin/sh

# Run the experiment in Slurm cluster:
# module load julia
# sh box_experiment_commandlist.sh > cmds.txt
# sbatch_commandlist -t 2:00:00 -mem 4000 -A project_2001274 -commands cmds.txt
# sbatch_commandlist -t 4:00:00 -mem 4000 -A project_2001274 -commands cmds.txt
# latter for N=512 case

CMD="julia"

box_width=0.1
box_potential=6.0
replications=10000
output_base="out/box2"

for res in Stratified StratifiedPartition StratifiedAlternate \
Systematic SystematicPartition SystematicAlternate \
SSP SSPPartition SSPAlternate \
SSS SSSPartition \
Killing Multinomial Residual; do
    for N in 64 128 256 512; do
          echo $CMD run_box_experiment.jl -N $N --resampling $res \
          --replications $replications --output ${output_base}_${res}_$N.jld2 \
          --box_width $box_width --box_potential $box_potential
    done
done
