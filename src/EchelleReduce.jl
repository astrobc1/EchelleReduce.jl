module EchelleReduce

using Reexport

include("recipe.jl")

include("tracing.jl")
@reexport using .Tracing

include("precalib.jl")
@reexport using .PreCalib

include("extract.jl")
@reexport using .Extract

include("reduction.jl")

include("optimal.jl")
@reexport using .OptimalExtraction

include("perfectionism.jl")
@reexport using .SPExtraction

end