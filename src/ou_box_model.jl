# Ornstein-Uhlenbeck model
include("ou_model.jl")

FKParticle = OUParticle

# This will be the "particle scratch", which will contain both model data & parameters, and which will be
# the 'scratch' argument of M_ar1!, lM_ar1, lG_sv
struct BoxParam
    box_centre::Float64
    box_width::Float64
    box_potential::Float64
end
BoxParam() = BoxParam(0.5,0.2,3.0)
struct FKScratch
    par::OUParam
    box_par::BoxParam
    FKScratch() = new(OUParam(),BoxParam())
end

# Box-shaped potential
function lG(k, x, scratch)
    x0 = scratch.box_par.box_centre; w = scratch.box_par.box_width
    V_bad = -scratch.par.Δ*scratch.box_par.box_potential
    V_good = 0.0
    abs(x.s-x0)<=w ? V_good : V_bad
end

