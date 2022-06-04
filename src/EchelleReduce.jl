module EchelleReduce

using Reexport

include("recipe.jl")

include("tracing.jl")
@reexport using .Tracing

@time begin
include("precalib.jl")
@reexport using .PreCalib
end

@time begin
include("extract.jl")
@reexport using .Extract
end

@time begin
include("reduction.jl")
end

@time begin
include("optimal.jl")
@reexport using .OptimalExtraction
end

@time begin
include("perfectionism.jl")
@reexport using .SPExtraction
end

end