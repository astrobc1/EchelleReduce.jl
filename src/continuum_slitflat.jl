export remove_slitflat_continuum

function remove_slitflat_continuum(
        image::Matrix{<:Real}, traces::OrderedDict{String, <:NamedTuple};
        xrange::Vector{Int},
        deg::Int=6,
    )

    # Image out
    image_out = copy(image)
    mask = falses(size(image))

    # dims
    ny, nx = size(image)

    # Continua
    continua = OrderedDict{String, Vector{Float64}}()

    # Arrs
    xarr = 1:nx
    yarr = 1:ny

    # Loop over traces/pixels
    for trace in values(traces)

        y = fill(NaN, nx)

        # Smooth cols
        for x in xrange[1]:xrange[2]
            inds = findall(trace.yc[x] + trace.yrange[1] .< yarr .< trace.yc[x] + trace.yrange[2])
            zcol = @views image[inds, x]
            y[x] = nanmedian(zcol)
        end

        # Fit
        s = quantile_filter(y, window=7)
        good = findall(isfinite.(s))
        pfit = Polynomials.fit(ArnoldiFit, good, s[good], deg)
        continua[trace.label] = pfit.(xarr)

        # Divide into image
        for x=xrange[1]:xrange[2]
            ymid = trace.yc[x]
            yi = max(Int(floor(ymid + trace.yrange[1])), 1)
            yf = min(Int(ceil(ymid + trace.yrange[2])), ny)
            mask[yi:yf, x] .= true
            image_out[yi:yf, x] ./= continua[trace.label][x]
        end

    end

    # Correct in between orders
    bad = findall(.~mask)
    image_out[bad] .= NaN

    # Return
    return image_out, continua

end