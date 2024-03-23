export extract_trace_optimal, OptimalExtractor, get_profile_spline


struct OptimalExtractor <: Extractor
    yc::Vector{Float64}
    trace_pos_deg::Union{Int, Nothing}
    xrange::Vector{Int}
    yrange::Vector{<:Real}
    yrange_extract::Union{Vector{Float64}, Nothing}
    yrange_extract_thresh::Real
    spl::Any
    profile_knot_spacing_x::Union{Real, Nothing}
    profile_knot_spacing_y::Union{Real, Nothing}
    profile_deg_x::Int
    profile_deg_y::Int
    remove_background::Bool
    background_smooth_width::Int
    max_iterations::Int
    read_noise::Float64
    badpix_nσ::Float64
end


function OptimalExtractor(;
        yc::Vector{Float64},
        trace_pos_deg::Union{Int, Nothing}=nothing,
        xrange::Vector{Int}, yrange::Union{Vector{<:Real}, Nothing},
        yrange_extract::Union{Vector{<:Real}, Nothing}, yrange_extract_thresh::Real=4,
        spl::Any=nothing,
        profile_knot_spacing_x::Union{Real, Nothing}=nothing, profile_knot_spacing_y::Union{Real, Nothing}=nothing,
        profile_deg_x::Int=2, profile_deg_y::Int=3,
        remove_background::Bool=false, background_smooth_width::Int=0,
        max_iterations::Int=20, read_noise::Real=0,
        badpix_nσ::Real=8
    )
    return OptimalExtractor(yc, trace_pos_deg, xrange, yrange, yrange_extract, yrange_extract_thresh, spl, profile_knot_spacing_x, profile_knot_spacing_y, profile_deg_x, profile_deg_y, remove_background, background_smooth_width, max_iterations, read_noise, badpix_nσ)
end

# OOP forward
function extract_trace(image::Matrix{<:Real}, opt::OptimalExtractor)
    return extract_trace_optimal(
        image;
        opt.yc, opt.trace_pos_deg,
        opt.xrange, opt.yrange, opt.yrange_extract, opt.yrange_extract_thresh,
        opt.profile_knot_spacing_x, opt.profile_knot_spacing_y,
        opt.profile_deg_x, opt.profile_deg_y,
        opt.remove_background, opt.background_smooth_width,
        opt.spl,
        opt.max_iterations,
        opt.read_noise,
        opt.badpix_nσ
    )
end

# Not OOP
function extract_trace_optimal(
        image::Matrix{<:Real};
        yc::Vector{Float64},
        trace_pos_deg::Union{Int, Nothing}=nothing,
        xrange::Vector{Int}, yrange::Vector{<:Real},
        spl::Any,
        profile_knot_spacing_x::Union{Real, Nothing}, profile_knot_spacing_y::Union{Real, Nothing}=nothing,
        profile_deg_x::Int=2, profile_deg_y::Int=3,
        yrange_extract::Union{Vector{<:Real}, Nothing}=nothing, yrange_extract_thresh::Real=4,
        remove_background::Bool=false, background_smooth_width::Int=0,
        max_iterations::Int=20, read_noise::Real=0,
        badpix_nσ::Real=8
    )

    @assert max_iterations > 0

    # Copy image
    image0 = image
    image = copy(image0)

    # Mask the trace
    mask_trace!(image, yc; yrange, xrange)
    image, yi = get_trace_image(image)
    yc = yc .- yi .+ 1

    # Initial opt spectrum and error
    spec1d, specerr1d = nothing, nothing

    # Repair the image
    image_rep = repair_bad_pix2d(image, yc; xrange, yrange)

    # Get new positions if desired
    if !isnothing(trace_pos_deg)
        image_rep_smooth = quantile_filter(image_rep, window=(3, 3))
        yc = get_trace_positions_centroids(image_rep_smooth, yc; xrange, yrange, deg=trace_pos_deg)
    end

    # Initial background
    if remove_background
        image_smooth = quantile_filter(image_rep, window=(3, 3))
        background = nanminimum(image_smooth, dim=1)
        background = quantile_filter(background, window=5)
        background_err = sqrt.(background)
        image_rep_nobg = image_rep .- transpose(background)
    else
        background, background_err = nothing, nothing
        image_rep_nobg = image_rep
    end

    # Get profile from positions
    if isnothing(spl)
        _spl = get_profile_spline(
            image_rep_nobg, yc, xrange, yrange;
            knot_spacing_x=profile_knot_spacing_x, knot_spacing_y=profile_knot_spacing_y,
            deg_x=profile_deg_x, deg_y=profile_deg_y
        )
    else
        _spl = spl
    end

    # Get aperture
    if isnothing(yrange_extract)
        _yrange_extract = get_extract_aperture(image, _spl; xrange, yrange, thresh=yrange_extract_thresh)
    else
        _yrange_extract = copy(yrange_extract)
    end

    # New background
    if remove_background
        background, background_err = compute_slit_background1d(image, yc; xrange, yrange=_yrange_extract)
        if background_smooth_width > 0
            if !isodd(background_smooth_width)
                background_smooth_width += 1
            end
            background .= quantile_filter(background, window=background_smooth_width)
            background_err .= quantile_filter(background_err, window=background_smooth_width)
        end

        # Get new profile
        if isnothing(spl)
            _spl = get_profile_spline(
                image_rep .- transpose(background), yc, xrange, yrange;
                knot_spacing_x=profile_knot_spacing_x, knot_spacing_y=profile_knot_spacing_y,
                deg_x=profile_deg_x, deg_y=profile_deg_y
            )
        else
            _spl = spl
        end

        # New aperture
        if isnothing(yrange_extract)
            _yrange_extract = get_extract_aperture(image, _spl; xrange, yrange, thresh=yrange_extract_thresh)
        else
            _yrange_extract = copy(yrange_extract)
        end

    end

    # Get P
    profile = eval_profile(image, yc, _spl; xrange, yrange)

    # Main loop
    for i=1:max_iterations

        # Print
        println("Iteration $i")

        # Do extraction
        spec1d, specerr1d = optimal_extraction(image, yc, profile; background, background_err, xrange, yrange=_yrange_extract, read_noise, n_iterations=3)

        # Look for outliers
        if i < max_iterations

            # Smooth 1D spectrum
            spec1d_smooth = quantile_filter(spec1d, window=3)

            # Reconvolve image into 2D space
            model_image_smooth = gen_model_image_background(image, spec1d_smooth, yc, profile; background, xrange, yrange, yrange_extract=_yrange_extract)

            # Current number of bad pixels
            n_bad_current = sum(isfinite.(image))

            # Flag
            flag_pixels2d!(image, model_image_smooth, read_noise, nσ=badpix_nσ)

            # New number
            n_bad_new = sum(isfinite.(image))
            
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
    model_image = gen_model_image_background(image, spec1d, yc, profile; background, xrange, yrange, yrange_extract=_yrange_extract)

    # Residuals
    residuals = image .- model_image

    # Out
    result = (;spec=spec1d, specerr=specerr1d, auxiliary=Dict{String, Any}("residuals" => residuals, "profile" => profile, "background" => background, "background_err" => background_err, "yi" => yi, "yc" => yc, "yrange_extract" => _yrange_extract, "yrange" => yrange))
    
    # Return
    return result

end


# Make a model image
function gen_model_image_background(image::Matrix{<:Real}, spec1d::Vector{Float64}, yc::Vector{<:Real}, profile::Matrix{<:Real}; xrange::Vector{Int}, yrange::Vector{<:Real}, yrange_extract::Vector{<:Real}, background::Union{Vector{<:Real}, Nothing}=nothing)

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
        ybottom_extract = ymid + yrange_extract[1]
        ytop_extract = ymid + yrange_extract[2]

        # Determine which pixels to use from the aperture in the fit
        window = Int(round(ybottom)):Int(ceil(ytop - 0.5))
        window_extract = Int(round(ybottom_extract)):Int(ceil(ytop_extract - 0.5))
        if window[1] < 1 || window[end] > ny
            continue
        end

        # Shift and normalize trace profile
        P = profile[window_extract, x]
        if any(.~isfinite.(P) .|| P .< 0)
            continue
        end
        P ./= sum(P)

        # Reconvolve
        model_image[window_extract, x] .= spec1d[x] .* P
        if !isnothing(background)
            model_image[window, x] .+= background[x]
        end

    end

    # Return
    return model_image

end


# Get extract aperture
function get_extract_aperture(image::Matrix{<:Real}, spl; xrange::Vector{Int}, yrange::Vector{<:Real}, thresh::Real=4)
    yarr0, profilec = eval_profile_coherent(image, spl; xrange, yrange)
    if length(size(profilec)) > 1
        profilec .-= transpose(nanminimum(profilec, dim=1))
        Pmed = nanmedian(profilec, dim=2)
    else
        profilec .-= nanminimum(profilec)
        Pmed = profilec
    end
    clamp!(Pmed, 0, Inf)
    Pmed ./= nanmaximum(Pmed)
    σ = sqrt(nansum(Pmed .* yarr0.^2) / nansum(Pmed))
    yi, yf = -thresh * σ, thresh * σ
    yi, yf = max(yi, yrange[1]), min(yf, yrange[2])
    #good_left = findall(yarr0 .< 0 .&& Pmed .> thresh)
    #good_right = findall(yarr0 .> 0 .&& Pmed .> thresh)
    #yi = length(good_left) == 0 ? yarr0[good_left[1]] : yarr0[1]
    #yf = length(good_right) == 0 ? yarr0[good_right[end]] : yarr0[end]
    #yi, yf = max(yi, yrange[1]), min(yf, yrange[2])
    return [yi, yf]
end

# Primary method once everything is known
function optimal_extraction(
        image::Matrix{<:Real}, yc::Vector{Float64}, profile::Matrix{<:Real};
        xrange::Vector{Int}, yrange::Vector{<:Real},
        background::Union{Vector{<:Real}, Nothing}=nothing,
        background_err::Union{Vector{<:Real}, Nothing}=nothing,
        read_noise::Real=0,
        n_iterations::Int=3
    )

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
            window = Int(round(ybottom)):Int(ceil(ytop - 0.5))

            # Mask for this window
            M = zeros(length(window))
            good = isfinite.(view(image, window, x))
            M[good] .= 1

            # Don't try to fit a single pixel
            if sum(M) <= 1
                continue
            end

            # Shift trace profile and normalize
            P = profile[window, x]
            if any(.~isfinite.(P) .|| P .< 0)
                continue
            end
            P ./= sum(P)

            # Data
            if !isnothing(background)
                S = image[window, x] .- background[x]
            else
                S = image[window, x]
            end

            # Flag negative flux
            bad = findall(S .<= 0)
            M[bad] .= 0
            S[bad] .= NaN

            # Check if column is worth extracting again
            if sum(M) <= 1
                continue
            end

            # Variance
            v = i == 1 ? copy(S) : spec1d[x] .* P
            v .+= !isnothing(background) ? background[x] .+ background_err[x]^2 : 0
            v .+= read_noise^2

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


function eval_profile_coherent(image::Matrix{<:Real}, spl::LSQBivariateSpline; xrange::Vector{Int}, yrange::Vector{<:Real})
    ny, nx = size(image)
    yi, yf = minimum(spl.y), maximum(spl.y)
    yarr0 = yi:yf
    P = fill(NaN, length(yarr0), nx)
    for x in xrange[1]:xrange[2]
        P[:, x] .= [exp(spl(x, y)[1]) for y in yarr0]
    end
    return yarr0, P
end

function eval_profile_coherent(image::Matrix{<:Real}, spl::Spline1D; xrange::Vector{Int}, yrange::Vector{<:Real})
    yi, yf = minimum(Dierckx.get_knots(spl)), maximum(Dierckx.get_knots(spl))
    yarr0 = yi:yf
    P = exp.(spl(yarr0))
    return yarr0, P
end


function eval_profile(image::Matrix{<:Real}, yc::Vector{Float64}, spl::LSQBivariateSpline; xrange::Vector{Int}, yrange::Vector{<:Real})
    ny, nx = size(image)
    P = fill(NaN, ny, nx)
    for x in xrange[1]:xrange[2]
        ymid = yc[x]
        ybottom = Int(round(ymid + yrange[1]))
        ytop = Int(ceil(ymid + yrange[2]))
        if (1 <= ybottom <= ny) && (1 <= ytop <= ny)
            window = ybottom:ytop
            P[window, x] .= [exp(spl(x, y)[1]) for y in window .- ymid]
        end
    end
    return P
end

function eval_profile(image::Matrix{<:Real}, yc::Vector{Float64}, spl::Spline1D; xrange::Vector{Int}, yrange::Vector{<:Real})
    ny, nx = size(image)
    P = fill(NaN, ny, nx)
    for x in xrange[1]:xrange[2]
        ymid = yc[x]
        ybottom = Int(round(ymid + yrange[1]))
        ytop = Int(ceil(ymid + yrange[2]))
        if (1 <= ybottom <= ny) && (1 <= ytop <= ny)
            window = ybottom:ytop
            P[window, x] .= [exp(spl(y)) for y in window .- ymid]
        end
    end
    return P
end

# Get profile
function get_profile_spline(
        image::Matrix{<:Real}, yc::Vector{Float64}, xrange::Vector{Int}, yrange::Vector{<:Real};
        knot_spacing_x::Real=100, knot_spacing_y::Real=1, deg_x::Int=3, deg_y::Int=3
    )

    # Image dims
    ny, nx = size(image)

    # Helpful arrays
    yarr = 1:ny

    # Vectors of coords
    xx = Float64[]
    yy = Float64[]
    zz = Float64[]
    sizehint!(xx, ny*nx)
    sizehint!(yy, ny*nx)
    sizehint!(zz, ny*nx)

    # begin
    #     xarr = 1:nx
    #     imshow(image, extent=(0.5, ny+.5, nx+.5, .5), vmin=0, vmax=10)
    #     plot(xarr, yc, c="red")
    #     ylim(0, ny)
    # end

    # Approx 1D spectrum
    spec1d_smooth = nansum(quantile_filter(image, window=(3, 3)), dim=1)
    spec1d_smooth .= quantile_filter(spec1d_smooth, window=3)
    spec1d_smooth ./= nanquantile(spec1d_smooth, 0.98)
    thresh = nanquantile(spec1d_smooth, 0.2)

    # Rectify and normalize but don't resample
    for x=xrange[1]:xrange[2]
        ymid = yc[x]
        ybottom = ymid + yrange[1]
        ytop = ymid + yrange[2]
        window = Int(round(ybottom)):Int(ceil(ytop - 0.5))
        if window[1] < 1 || window[end] > ny || spec1d_smooth[x] < thresh
            continue
        end
        coly = yarr[window] .- ymid
        colz = image[window, x]
        bad = findall(colz .< 0)
        colz[bad] .= NaN
        colz ./= nansum(colz)
        for j in eachindex(coly)
            push!(xx, x)
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

    # Get knots for crude spline
    _, knots_y = get_knots(xx, yy; deg_x=0, deg_y, knot_spacing_x=0, knot_spacing_y)

    # Crude spline
    spl0 = Dierckx.Spline1D(yy, logzz, knots_y)

    # Spline
    Pflat = exp.(spl0(yy))
    norm_res = (Pflat .- zz) ./ sqrt.(Pflat)
    good = findall(norm_res .< 4 * 1.4826 * nanmad(norm_res))
    xx, yy, zz, logzz = xx[good], yy[good], zz[good], logzz[good]

    # Artificially extend data to prevent spline boundary problems
    for x in xrange[1]:xrange[2]

        # Indices
        inds = findall(xx .== x)
        if length(inds) == 0
            continue
        end
        ys = @view yy[inds]
        k1, k2 = argmin(ys), argmax(ys)
        yi, yf = ys[k1], ys[k2]

        # Lower
        push!(xx, x)
        push!(yy, yi-1)
        push!(zz, zz[k1])
        push!(logzz, logzz[k1])

        push!(xx, x)
        push!(yy, yi-2)
        push!(zz, zz[k1])
        push!(logzz, logzz[k1])

        # Upper
        push!(xx, x)
        push!(yy, yf+1)
        push!(zz, zz[k2])
        push!(logzz, logzz[k2])

        push!(xx, x)
        push!(yy, yf+2)
        push!(zz, zz[k2])
        push!(logzz, logzz[k2])

    end

    # Sort according to y
    ss = sortperm(yy)
    xx .= xx[ss]
    yy .= yy[ss]
    zz .= zz[ss]
    logzz .= logzz[ss]

    # Get knots
    knots_x, knots_y = get_knots(xx, yy; deg_x, deg_y, knot_spacing_x, knot_spacing_y)

    # Get spline
    if deg_x == 0
        spl = Dierckx.Spline1D(yy, logzz, knots_y)
    else
        spl = LSQBivariateSpline(xx, yy, logzz, knots_x, knots_y; kx=deg_x, ky=deg_y)
    end

    # Return
    return spl
end


function get_knots(
        x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
        deg_x::Int, deg_y::Int, knot_spacing_x::Real, knot_spacing_y::Real, pad::Real=1E-2
    )
    if deg_x > 0
        knots_x = collect(range(minimum(x) + pad, step=knot_spacing_x, stop=maximum(x) - pad))
    else
        knots_x = nothing
    end
    knots_y = collect(range(minimum(y) + pad, step=knot_spacing_y, stop=maximum(y) - pad))
    return knots_x, knots_y
end