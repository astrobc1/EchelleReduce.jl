module Tracing

using Statistics
using EchelleBase

export trace

function trace end

"""
    gen_trace_image(traces::Vector, ny, nx, sregion::SpecRegion2d)
Generate an nx x ny image with values according to the labels in each trace dictionary.
- `traces` The vector of trace parameters (dictionaries)
- `ny`. The number of vertical pixels.
- `nx`. The number of horizontal pixels.
- `sregion`. The spectral region to further bound the left and right ends of the image.
"""
function gen_trace_image(traces::Vector, ny, nx, sregion::SpecRegion2d)

    # Initiate order image
    order_image = fill(NaN, (ny, nx))

    # Helpful arr
    xarr = [1:nx;]

    for trace ∈ traces
        order_center = trace["poly"].(xarr)
        for x=1:nx
            ymid = order_center[x]
            y_low = Int(floor(ymid - trace["height"] / 2))
            y_high = Int(ceil(ymid + trace["height"] / 2))
            if y_low < 1 || y_high > nx
                continue
            end
            order_image[y_low:y_high, x] .= parse(Float64, trace["label"])
        end
    end

    # Mask image
    mask!(order_image, sregion)

    # Return
    return order_image
end


function group_peaks(x; sep)
    peak_centers = Float64[]
    heights = Float64[]
    prev_i = 1
    for i=1:length(x) - 1
        if x[i+1] - x[i] > sep
            push!(peak_centers, mean(x[prev_i:i]))
            push!(heights, x[i] - x[prev_i])
            prev_i = i + 1
        end
    end
    push!(peak_centers, mean(x[prev_i:end]))
    push!(heights, x[end] - x[prev_i])
    return peak_centers, heights
end

# Default tracing alg
include("peak_tracer.jl")
            
end