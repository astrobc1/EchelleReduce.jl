export Boxcar

struct Boxcar <: Extractor
    xrange::Vector{Int}
    yrange::Vector{Float64}
    yc::Vector{Float64}
    collapse_function::String
    read_noise::Real
end

function Boxcar(;
        xrange::Vector{Int},
        yrange::Vector{<:Real},
        yc::Vector{Float64},
        collapse_function::String,
        read_noise::Real=0,
    )
    return Boxcar(xrange, Float64.(yrange), yc, collapse_function, read_noise)
end

function extract_trace(image::Matrix{<:Real}, extractor::Boxcar)
    return extract_boxcar(
        image;
        extractor.xrange,
        extractor.yrange,
        extractor.yc,
        extractor.collapse_function,
        extractor.read_noise
    )
end


function extract_boxcar(
        image::Matrix{<:Real};
        xrange::Vector{Int},
        yrange::Vector{Float64},
        yc::Vector{Float64},
        collapse_function::String,
        read_noise::Real=0
    )

    collapse_function = lowercase(collapse_function)
    if collapse_function == "mean"
        _collapse_function = nanmean
    elseif collapse_function == "median"
        _collapse_function = nanmedian
    else
        error("collapse_function must be mean or median, got $(collapse_function)")
    end

    ny, nx = size(image)
    spec = fill(NaN, nx)
    specerr = fill(NaN, nx)
    A = yrange[2] - yrange[1] + 1
    for x=xrange[1]:xrange[2]
        ymid = yc[x]
        ybottom = ymid + yrange[1]
        ytop = ymid + yrange[2]
        if ybottom < 1 || ytop > ny
            continue
        end
        window = Int(round(ybottom)):Int(ceil(ytop - 0.5))
        S = @views image[window, x]
        spec[x] = _collapse_function(S) * A
        specerr[x] = nansum(1 ./ (S .+ read_noise^2).^2)^-0.5
    end

    return (;spec, specerr, auxiliary=Dict{String, Any}())

end