export extract_spline2d

function extract_spline2d(
    image::Matrix{<:Real}, yc::Vector{<:Real}, spl::LSQBivariateSpline;
    xrange::Union{Vector{Int}, Nothing}=nothing,
    yrange::Union{Vector{<:Real}, Nothing},
    max_iterations::Int=20,
    read_noise::Real=0)

    # Copy image
    image = copy(image)

    # Dims
    ny, nx = size(image)

    # Xrange
    if isnothing(xrange)
        good = findall(isfinite.(image))
        goodx = get_inds1d(good, dim=1)
        xrange = [minimum(goodx), maximum(goodx)]
    end

    # Initial opt spectrum and error
    spec1d, specerr1d = fill(NaN, nx), fill(NaN, nx)

    # Main loop
    for i=1:max_iterations

        # Print
        println("Iteration $i")

        # Do extraction
        r = extract_spline2d_alg(trace_image, yc, spl, xrange, yrange, read_noise, 3)
        spec1d .= r[1]
        specerr1d .= r[2]

        # Look for outliers
        if i < max_iterations

            # Smooth 1D spectrum
            spec1d_smooth = quantile_filter1d(spec1d, width=3)

            # Reconvolve image into 2D space
            model_image_smooth = gen_model_image(trace_image, spec1d_smooth, yc, spl, xrange, yrange)

            # Current number of bad pixels
            n_bad_current = sum(isfinite.(trace_image))

            # Flag
            flag_pixels2d!(trace_image, model_image_smooth, read_noise, nσ=badpix_σ)

            # New number
            n_bad_new = sum(isfinite.(trace_image))
            
            # Break if nothing new is flagged
            if n_bad_current == n_bad_new && i > 1
                break
            end
        end
    end

    # Sync any bad pixels
    bad = findall(.~isfinite.(spec1d) .|| (spec1d .<= 0) .|| .~isfinite.(specerr1d) .|| (specerr1d .<= 0))
    spec1d[bad] .= NaN
    specerr1d[bad] .= NaN

    # Generate the final 2D model
    model_image = gen_model_image(trace_image, spec1d, yc, spl, xrange, yrange)

    # Out
    result = (;spec1d, specerr1d, model_image, spl)
    
    # Return
    return result
end

function extract_spline2d(
    image::Matrix{<:Real};
    xrange::Union{Vector{Int}, Nothing}=nothing,
    yrange::Union{Vector{<:Real}, Nothing},
    max_iterations::Int=20,
    read_noise::Real=0)

    # Copy image
    image = copy(image)

    # Dims
    ny, nx = size(image)

    # Xrange
    if isnothing(xrange)
        good = findall(isfinite.(image))
        goodx = get_inds1d(good, dim=1)
        xrange = [minimum(goodx), maximum(goodx)]
    end

    # Spl
    spl = nothing

    # Initial opt spectrum and error
    spec1d, specerr1d = fill(NaN, nx), fill(NaN, nx)

    # Main loop
    for i=1:max_iterations

        # Print
        println("Iteration $i")

        # Get positions
        yc = get_trace_centroids()

        # Get weights
        spl = get_extraction_weights(
            image, yc, xrange, yrange;
            knot_spacing_x, knot_spacing_y, degx, degy
        )

        # Do extraction
        r = extract_spline2d_alg(trace_image, yc, spl, xrange, yrange, read_noise, 3)
        spec1d .= r[1]
        specerr1d .= r[2]

        # Look for outliers
        if i < max_iterations

            # Smooth 1D spectrum
            spec1d_smooth = quantile_filter1d(spec1d, width=3)

            # Reconvolve image into 2D space
            model_image_smooth = gen_model_image(trace_image, spec1d_smooth, yc, spl, xrange, yrange)

            # Current number of bad pixels
            n_good_current = sum(isfinite.(trace_image))

            # Flag
            flag_pixels2d!(trace_image, model_image_smooth, read_noise, nσ=badpix_σ)

            # New number
            n_good_new = sum(isfinite.(trace_image))
            
            # Break if nothing new is flagged
            if n_good_current == n_good_new && i > 1
                break
            end
        end
    end

    # Sync any bad pixels
    bad = findall(.~isfinite.(spec1d) .|| (spec1d .<= 0) .|| .~isfinite.(specerr1d) .|| (specerr1d .<= 0))
    spec1d[bad] .= NaN
    specerr1d[bad] .= NaN

    # Generate the final 2D model
    model_image = gen_model_image(trace_image, spec1d, yc, spl, xrange, yrange)

    # Out
    result = (;spec1d, specerr1d, model_image, spl)
    
    # Return
    return result
end


function extract_spline2d_alg(
    image::Matrix{<:Real},
    yc::Vector{<:Real},
    spl::LSQBivariateSpline,
    xrange::Vector{Int},
    yrange::Vector{<:Real},
    read_noise::Real=0
    n_iterations::Int=3)

    # Image dims
    ny, nx = size(image)

    # Outputs
    spec1d = fill(NaN, nx)
    specerr1d = fill(NaN, nx)

    # Do a few iterations to get better estimate for variance
    for i=1:n_iterations

        # Loop over cols
        for x=xrange[1]:xrange[2]

            # Middle, lower, upper bound
            ymid = yc[x]
            ybottom = ymid + yrange[1]
            ytop = ymid + yrange[2]

            # Continue if out of frame
            if ybottom < 1 || ytop > ny
                continue
            end

            # Determine which pixels to use from the aperture in the fit
            window = Int(round(ybottom)):Int(round_half_down(ytop))

            # Mask for this window
            M = zeros(length(window))
            good = isfinite.(view(trace_image, window, x))
            M[good] .= 1

            # Don't try to fit a single pixel
            if sum(M) <= 1
                continue
            end

            # Shift trace profile and normalize
            P = exp.(vec(trace_profile(x, yarr[window] .- ymid)))
            if any(.~isfinite.(P) .|| P .< 0)
                continue
            end
            P ./= sum(P)

            # Data
            S = trace_image[window, x]

            # Flag negative flux
            bad = findall(S .<= 0)
            M[bad] .= 0
            S[bad] .= NaN

            # Check if column is worth extracting again
            if sum(M) <= 1
                continue
            end

            # Variance
            if i == 1
                v = read_noise^2 .+ S
            else
                v = read_noise^2 .+ spec1d[x] .* P
            end

            # Optimal weights
            w = M ./ v

            # Least squares
            spec1d[x] = nansum(w .* S .* P) / nansum(w .* P.^2)

            # Horne variance
            specerr1d[x] = sqrt(nansum(M .* P) / nansum(w .* P.^2))
        
        end

    end

    return spec1d, specerr1d
end


function gen_model_image(image::Matrix{<:Real}, yc::Vector{<:Real}, spl::LSQBivariateSpline, xrange::Vector{Int}, yrange::Vector{<:Real})

    # Dims
    ny, nx = size(image)

    # Initialize model
    model_image = fill(NaN, (ny, nx))

    # Loop over cols
    for x=1:nx

        # Centroid, lower, upper bound
        ymid = yc[x]
        ybottom = ymid + yrange[1]
        ytop = ymid + yrange[2]
        if ybottom < 1 || ytop > ny
            continue
        end

        # Determine which pixels to use from the aperture in the fit
        window = Int(round(ybottom)):Int(round_half_down(ytop))
        if window[1] < 1 || window[end] > ny
            continue
        end

        # Shift and normalize trace profile
        P = exp.(vec(spl(x, yarr[window] .- ymid)))
        if any(.~isfinite.(P) .|| P .< 0)
            continue
        end
        P ./= sum(P)

        # Reconvolve
        model_image[window, x] .= spec1d[x] .* P

    end

    # Return
    return model_image

end


function flag_pixels2d!(image::Matrix{<:Real}, model_image::Matrix{<:Real}, read_noise::Real; nσ::Real)

    # Smooth 2D model for variance estimate
    model_image_smooth = quantile_filter2d(model_image, width=3)

    # Normalized residuals
    norm_res = (image .- model_image) ./ sqrt.(model_image_smooth .+ read_noise^2)

    # Which pixels to use
    use = findall(.~(norm_res .== 0) .&& isfinite.(norm_res) .&& (model2d .!= 0))
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



function get_extraction_weights(
    image::Matrix{<:Real}, yc::Weights, xrange::Vector{Int}, yrange::Vector{<:Real};
    knot_spacing_x::Real=100, knot_spacing_y::Real=1, degx::Int=3, degy::Int=3)

    # Image dims
    ny, nx = size(image)
    
    # Helpful arrays
    xarr = [1:nx;]
    yarr = [1:ny;]

    # Vectors of coords
    xx = Float64[]
    yy = Float64[]
    zz = Float64[]
    sizehint!(xx, ny*nx)
    sizehint!(yy, ny*nx)
    sizehint!(zz, ny*nx)
    
    # Rectify and normalize but don't resample
    for i=xrange[1]:xrange[2]
        good = @views findall(isfinite.(image[:, i]))
        if length(good) <= 3
            continue
        end
        ymid = yc[x]
        ybottom = ymid + yrange[1]
        ytop = ymid + yrange[2]
        window = Int(round(ybottom)):Int(round_half_down(ytop))
        if window[1] < 1 || window[end] > ny
            continue
        end
        coly = yarr[window] .- ymid
        colz = image[window, i]
        bad = findall(colz .< 0)
        colz[bad] .= NaN
        colz ./= nansum(colz)
        for j=1:ny
            push!(xx, i)
            push!(yy, coly[j])
            push!(zz, colz[j])
        end
    end

    # Change to logz
    logzz = log.(zz)
    
    # Remove bad
    good = findall(isfinite.(logzz))
    xx, yy, zz, logzz = xx[good], yy[good], zz[good], logzz[good]

    # Sort according to y
    ss = sortperm(yy)
    xx .= xx[ss]
    yy .= yy[ss]
    zz .= zz[ss]
    logzz .= logzz[ss]

    # # Flag deviations from median
    splmed = CubicSpline(tplogz, tpy)
    P = exp.(splmed.(yy))
    norm_res = (tphr .- zz) ./ sqrt.(tphr)
    σ = 1.4826 * nanmad(norm_res)
    good = findall(abs.(norm_res) .< 4 * σ)
    xx, yy, zz, logzz = xx[good], yy[good], zz[good], logzz[good]

    # Get knots
    pad = 1E-2
    knotsx = collect(range(minimum(xx) + pad, step=knot_spacing_x, stop=maximum(xx) - pad))
    knotsy = collect(range(minimum(yy) + pad, step=knot_spacing_y, stop=maximum(yy) - pad))

    # Spline
    spl = LSQBivariateSpline(xx, yy, logzz, knotsx, knotsy; kx=degx, ky=degy)

    # Return
    return spl
end