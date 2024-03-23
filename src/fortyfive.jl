export FortyFiveExtractor

struct FortyFiveExtractor <: Extractor
    yc::Vector{Float64}
    profile::Matrix{Float64}
    xrange::Vector{Int}
    yrange::Vector{<:Real}
    max_iterations::Int
    read_noise::Float64
    badpix_nσ::Float64
end


function FortyFiveExtractor(;
        yc::Vector{Float64},
        trace_pos_deg::Union{Int, Nothing}=nothing,
        xrange::Vector{Int}, yrange::Union{Vector{<:Real}, Nothing},
        yrange_extract::Union{Vector{<:Real}, Nothing}, yrange_extract_thresh::Real=0.015,
        spl::Any=nothing,
        profile_knot_spacing_x::Union{Real, Nothing}=nothing, profile_knot_spacing_y::Union{Real, Nothing}=nothing,
        profile_deg_x::Int=2, profile_deg_y::Int=3,
        remove_background::Bool=false, background_smooth_width::Int=0,
        max_iterations::Int=20, read_noise::Real=0,
        badpix_nσ::Real=8
    )
    return FortyFiveExtractor(yc, trace_pos_deg, xrange, yrange, yrange_extract, yrange_extract_thresh, spl, profile_knot_spacing_x, profile_knot_spacing_y, profile_deg_x, profile_deg_y, remove_background, background_smooth_width, max_iterations, read_noise, badpix_nσ)
end

# OOP forward
function extract_trace(image::Matrix{<:Real}, opt::FortyFiveExtractor)
    return extract_trace_fortyfive(
        image;
        opt.weights,
        opt.yc, opt.trace_pos_deg,
        opt.xrange, opt.yrange,
        opt.max_iterations,
        opt.read_noise,
        opt.badpix_nσ
    )
end

# Not OOP
function extract_trace_fortyfive(
        image::Matrix{<:Real};
        yc::Vector{Float64},
        trace_pos_deg::Union{Int, Nothing}=nothing,
        xrange::Vector{Int}, yrange::Vector{<:Real},
        spl::Any,
        profile_knot_spacing_x::Union{Real, Nothing}, profile_knot_spacing_y::Union{Real, Nothing}=nothing,
        profile_deg_x::Int=2, profile_deg_y::Int=3,
        yrange_extract::Union{Vector{<:Real}, Nothing}=nothing, yrange_extract_thresh::Real=0.02,
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
        spec1d, specerr1d = fortyfive_extraction(image, yc, profile; background, background_err, xrange, yrange=_yrange_extract, read_noise, n_iterations=3)

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


# Primary method once everything is known
function fortyfive_extraction(
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
            if ybottom < 1 || ytop > ny || !(x - yrange[1] < x < x + yrange[1])
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