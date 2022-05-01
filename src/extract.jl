module Extract

using Reexport

export SpectralExtractor

abstract type SpectralExtractor end

include("extraction.jl")
include("extraction_utils.jl")

end