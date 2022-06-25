module Extract

using Reexport

export SpectralExtractor

"""
    SpectralExtractor
The base type for a spectral extraction algorithm.
"""
abstract type SpectralExtractor end

include("extraction.jl")
include("extraction_utils.jl")
include("plotting.jl")

end