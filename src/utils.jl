
function get_trace_positions_centroids(
        image::Matrix{<:Real}, yc::Vector{Float64};
        yrange::Vector{<:Real}, xrange::Vector{Int}, deg::Int, n_iterations::Int=1,
    )

    # The image dimensions
    ny, nx = size(image)

    # Y centroids
    ycens = fill(NaN, nx)

    for _=1:n_iterations

        # Loop over cols
        for x=xrange[1]:xrange[2]
            ymid = yc[x]
            ybottom = max(Int(floor(ymid + yrange[1] + 1)), 1)
            ytop = min(Int(ceil(ymid + yrange[2] - 1)), ny)
            if ybottom < 1 || ytop > ny
                continue
            end
            col = view(image, ybottom:ytop, x)
            good = findall(isfinite.(col))
            if length(good) <= 1
                continue
            end
            ycens[x] = @views sum(good .* col[good]) / sum(col[good]) + ybottom - 1
        end

    end

    # Flag
    ycens_smooth = quantile_filter(ycens, window=3)
    bad = findall(abs.(ycens - ycens_smooth) .> 0.5)
    ycens[bad] .= NaN

    # Fit
    good = findall(isfinite.(ycens))
    pfit, _ = polyfit1d(good, ycens[good]; deg, max_iterations=5, nσ=4)

    # Out
    yc_out = pfit.(1:nx)

    # Return
    return yc_out

end


function compute_slit_background1d(
        image::Matrix{<:Real}, yc::Vector{Float64};
        yrange::Vector{<:Real}, xrange::Vector{<:Real}
    )

    # Numbers
    ny, nx = size(image)
    yarr = 1:ny

    # Vectors
    background = fill(NaN, nx)
    background_err = fill(NaN, nx)

    for x=xrange[1]:xrange[2]

        # Bounds
        ymid = yc[x]
        ybottom = floor(ymid + yrange[1])
        ytop = ceil(ymid + yrange[2])

        # Sanity check
        if ybottom < 1 || ytop > ny
            continue
        end

        # Which pixels to use
        use = findall(isfinite.(view(image, :, x)) .&& (yarr .< ybottom .|| yarr .> ytop))
        n_good = length(use)

        # Cases
        if n_good > 2
            background[x] = @views nanmedian(image[use, x])
            background_err[x] = sqrt(background[x] ./ (n_good - 1))
        else
            good = @views findall(isfinite.(image[:, x]))
            if length(good) > 1
                S = image[good, x]
                ss = sortperm(S)
                S .= S[ss]
                background[x] = @views nanmean(S[1:2])
                background_err[x] = @views sqrt.(background[x])
            else
                background[x] = @views nanminimum(image[:, x])
                background_err[x] = sqrt(background[x])
            end
        end
    end

    # Return
    return background, background_err

end


function flag_pixels2d!(image::Matrix{<:Real}, model_image::Matrix{<:Real}, read_noise::Real; nσ::Real)

    # Smooth 2D model for variance estimate
    model_image_smooth = quantile_filter(model_image, window=(3, 3))

    # Normalized residuals
    norm_res = (image .- model_image) ./ sqrt.(model_image_smooth .+ read_noise^2)

    # Which pixels to use
    use = findall(.~(norm_res .== 0) .&& isfinite.(norm_res) .&& (model_image .!= 0))
    n_use = length(use)

    # Compute new normalized rms
    rms = @views sqrt(nansum(norm_res[use].^2) / n_use)

    # Find bad
    bad = findall(abs.(norm_res) .> nσ * rms)

    # Mask
    image[bad] .= NaN

    # Return
    return image
end


function mask_trace!(image::Matrix{<:Real}, yc::Vector{Float64}; xrange::Union{Vector{Int}, Nothing}, yrange::Union{Vector{<:Real}, Nothing})

    # Image dims
    ny, nx = size(image)

    # Top/Bottom
    if !isnothing(yrange)
        for x=1:nx
            ymid = yc[x]
            ybottom = Int(floor(ymid + yrange[1]))
            ytop = Int(ceil(ymid + yrange[2]))
            if ybottom > 1 && ybottom < ny
                image[1:ybottom-1, x] .= NaN
            end
            if ytop + 1 > 1 && ytop + 1 < ny
                image[ytop+1:end, x] .= NaN
            end
            if ybottom < 1 || ytop > ny
                image[:, x] .= NaN
            end
        end
    end

    # Left/Right
    if !isnothing(xrange)
        xi, xf = xrange
        if xi > 1
            image[:, 1:xi] .= NaN
        end
        if xf < nx
            image[:, nx+1:end] .= NaN
        end
    end

    # Return
    return image
end


function get_trace_image(image::Matrix{<:Real})
    ny = size(image, 1)
    good = findall(isfinite.(image))
    goody = get_inds1d(good, dim=1)
    yi, yf = minimum(goody), maximum(goody)
    yi, yf = max(yi - 1, 1), min(yf + 1, ny)
    trace_image = image[yi:yf, :]
    return trace_image, yi
end