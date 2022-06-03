using Pkg
Pkg.activate("..")

using ArgParse, JLD2

include("gen_smc.jl")
include("ou_box_model.jl")
include("smoothing_test.jl")

# Discretisation sizes
Δs = 2.0 .^ (-(0:2:12))

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--resampling", "-r"
            help = "Resampling scheme"
            arg_type = String
            default = "Killing"
        "--adaptive_threshold", "-a"
            help = "Adaptive resampling threshold (<1.0)"
            arg_type = Float64
            default = 1.0
        "--num_particles", "-N"
            help = "Number of particles"
            arg_type = Int
            default = 128
        "--time_physical", "-T"
            help = "Physical time of experiment"
            arg_type = Int
            default = 5
        "--replications", "-m"
            arg_type = Int
            default = 100
        "--box_centre"
            arg_type = Float64
            default = 0.5
        "--box_width"
            arg_type = Float64
            default = 0.2
        "--box_potential"
            arg_type = Float64
            default = 3.0
        "--output", "-o"
            arg_type = String
            help = "File name where output is stored"
            required = true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
# "Physical time" of the scenario
T_phys = parsed_args["time_physical"]
N = parsed_args["num_particles"]
rep = parsed_args["replications"]
out = parsed_args["output"]
box_centre = parsed_args["box_centre"]
box_width = parsed_args["box_width"]
box_potential = parsed_args["box_potential"]
adaptive_threshold = parsed_args["adaptive_threshold"]
adaptive_threshold = (adaptive_threshold >= 1.0) ? nothing : adaptive_threshold

AllResamplings = resampling_aliases(N)
res = AllResamplings[Symbol(parsed_args["resampling"])]

BoxParam() = BoxParam(box_centre, box_width, box_potential)

L, X = single_smoothing_experiment(res, Δs, T_phys, rep, N,
    FKParticle, FKScratch, M_ar1!, lG; set_delta! = set_delta!, 
    adaptive_threshold = adaptive_threshold)

jldsave(out; L, X)