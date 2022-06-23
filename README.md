# Weakly informative resampling codes

Source codes for the experiments in:

> N. Chopin, S. S. Singh, T. Soto and M. Vihola. **On resampling schemes for particle filters with weakly informative observations**.
[arXiv:2203.10037](http://arxiv.org/abs/2203.10037)

## Setting up

Start by installing a fresh copy of [Julia](https://julialang.org/downloads/). The experiments were run with Julia 1.7.1, which is recommended for reproducibility. (You can also use your existing Julia installation, and the  codes should work with other than Julia 1.7.1. too.)

Get a local copy of the codes and start julia by running the following commends (in a shell):
```shell
git clone https://github.com/mvihola/weakly-informative-resampling-codes
cd weakly-informative-resampling-codes
julia
```
Then, in the Julia REPL, do:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
This should install all the required packages for you.

## How to inspect the results of the experiments

You may produce the plots shown in the paper from pre-calculated summaries as follows:
```julia
include("src/box2_experiment_plots.jl")
include("src/box2_adaptive_experiment_plots.jl")
include("src/luminosity_experiment_plots.jl")
```
The scripts produce more plots than those reported in the paper.

## How to re-run the experiments

The experiments were designed to be run in a cluster, and for that reason, there are shell scripts which generate command lines, which can be executed independently in parallel.

Run the shell scripts (in `src` folder):
```shell
sh box2_experiment_commandlist.sh > cmds_box2.txt
sh box2_adaptive_experiment.commandlist.sh > cmds_box2_adaptive.txt
sh luminosity_experiment_commandlist.sh > cmds_luminosity.txt
```
This creates three text files, with all commands that run the experiments reported in Section 7.1, 7.2 and 7.3, respectively.

In principle, it should be possible to run all the commands in a single file at once, of instance:
```shell
sh cmds_box2.txt
```
would run all the experiments in Section 7.1.  Because there are many experiments, running all the experiments like this takes a lot of time.

We ran the experiments instead in [CSC](https://www.csc.fi/en/home) Slurm cluster, using the following command:
```shell
sbatch_commandlist -t 4:00:00 -mem 4000 -commands cmds_box2.txt
```

After running the files, you may run the following in the Julia REPL:
```julia
include("src/box2_experiment_gather.jl")
include("src/box2_adaptive_experiment_gather.jl")
include("src/luminosity_experiment_gather.jl")
```
These commands create the files that were visualised above.

## Contents

Here is a short description of the code files in `src` folder:

### Generic codes

* `gen_smc.jl`: A 'generic' particle filter implementation, inspired by [SequentialMonteCarlo.jl](https://github.com/awllee/SequentialMonteCarlo.jl), which accomodates a variety of resampling algorithms
* `resampling.jl`: The resampling algorithms that can be used with `gen_smc.jl`

### Experiments in Sections 7.1 and 7.2

* `out/box2_experiment_summaries.jld2`: Saved summaries of experiments in Section 7.1
* `out/box2_adaptive_experiment_summaries.jld2`: Saved summaries of experiments in Section 7.2
* `box2_experiment_*.jl`: Experiments of Section 7.1
* `box2_adaptive_experiment_*.jl`: Experiments of Section 7.2
* `run_box_experiment.jl`: Run an individual experiment
* `ou_box_model.jl` (and `ou_model.jl`): Defintion of the Ornstein-Uhlenbeck model with box potential model.
* `smoothing_test.jl` and `smoothing_test_analysis.jl`: Codes for doing a generic normalising constant/filtering/smoothing experiment, for calculating summaries and plotting the summaries

### Experiments in Sections 7.3

* `luminosity_data.jld2`: The synthetic data used in the experiment
* `out/luminosity_single_experiment.jld2`: Run single experiment and plot Figure 7 
* `out/luminosity_experiment_summaries.jld2`: Saved summaries of experiments in Section 7.3
* `luminosity_experiment_*.jl`: The experiments
* `run_luminosity_experiment.jl`: Run an individual experiment
* `random_walk_poissn.jl`: Definition of the model
* `generate_poisson_data.jl`: Fnctions for generating the synthetic data 
