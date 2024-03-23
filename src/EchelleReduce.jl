module EchelleReduce

using PyCall, PyPlot
using NaNStatistics
using OrderedCollections
using LsqFit
using JLD2
using Infiltrator
using SpecialFunctions
using Dierckx
using Polynomials

using Pkg
Pkg.develop(path="/Users/cale/Codes/JuliaProjects/Echelle/")
using Echelle

include("utils.jl")
include("interp2d.jl")

include("tracing.jl")
include("tracing_boxcar.jl")
include("tracing_gauss.jl")
include("calibration.jl")

include("continuum_slitflat.jl")

include("extraction.jl")
include("spline2d.jl")
include("boxcar.jl")
include("optimal.jl")

include("plotting.jl")


end
