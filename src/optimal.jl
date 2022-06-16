module OptimalExtraction

import DataInterpolations
import Polynomials
using NaNStatistics
using Infiltrator
using EchelleBase
using EchelleReduce

export OptimalExtractor, optimal_extraction

mutable struct OptimalExtractor <: SpectralExtractor
    n_iterations::Int
    remove_background::Bool
    background_smooth_width::Int
    oversample_profile::Int
    trace_pos_deg::Int
    badpix_σ::Float64
    extract_aperture::Union{Vector{Int}, Nothing}
end

function OptimalExtractor(;n_iterations=20, remove_background=false, background_smooth_width=nothing, oversample_profile=8, trace_pos_deg=4, badpix_σ=6, extract_aperture=nothing)
    if isnothing(background_smooth_width)
        background_smooth_width = 0
    end
    return OptimalExtractor(n_iterations, remove_background, background_smooth_width, oversample_profile, trace_pos_deg, badpix_σ, extract_aperture)
end

function Extract.extract_trace(extractor::OptimalExtractor, image, sregion, trace_params; badpix_mask=nothing, read_noise=0.0)

    # Copy image
    image = copy(image)

    # Full dims
    ny, nx = size(image)

    # Initiate mask
    if isnothing(badpix_mask)
        badpix_mask = ones(ny, nx)
    else
        badpix_mask = copy(badpix_mask)
    end

    # Refine initial window
    trace_positions = Extract.refine_initial_trace_window(image, badpix_mask, sregion, trace_params, n_iterations=3)

    # Mask image based on trace aperture
    trace_image = copy(image)
    trace_mask = copy(badpix_mask)
    for x=1:nx
        ymid = trace_positions(x)
        y_low = Int(floor(ymid - trace_params["height"] / 2))
        y_high = Int(ceil(ymid + trace_params["height"] / 2))
        if y_low > 1 && y_low < ny
            trace_image[1:y_low-1, x] .= NaN
        else
            trace_image[:, x] .= NaN
        end
        if y_high > 1 && y_high < ny
            trace_image[y_high+1:end, x] .= NaN
        else
            trace_image[:, x] .= NaN
        end
    end

    # Mask image ends
    trace_image[:, 1:sregion.pixmin] .= NaN
    trace_image[:, sregion.pixmax:end] .= NaN

    # Sync
    bad = findall(.~isfinite.(trace_image) .|| (trace_mask == 0))
    trace_image[bad] .= NaN
    trace_mask[bad] .= 0

    # Crop in the y direction
    good = findall(isfinite.(trace_image))
    goody = [coord.I[1] for coord ∈ good]
    yi, yf = minimum(goody), maximum(goody)
    trace_image = @view trace_image[yi:yf, :]
    trace_mask = @view trace_mask[yi:yf, :]
    ny, nx = size(trace_image)
    trace_positions -= yi

    # Flag obvious bad pixels again
    trace_image_smooth = maths.median_filter2d(trace_image, 5)
    peak = maths.weighted_median(trace_image_smooth, p=0.99)
    bad = findall((trace_image .< 0) .|| (trace_image .> 50 * peak))
    trace_image[bad] .= NaN
    trace_mask[bad] .= 0

    # Estimate background
    if extractor.remove_background
        background = nanminimum(trace_image, dim=1)
        background_err = sqrt.(background)
    else
        background, background_err = nothing, nothing
    end

    if extractor.remove_background
        trace_image_no_background = trace_image .- background'
    else
        trace_image_no_background = trace_image
    end

    trace_positions = Extract.compute_trace_positions_centroids(trace_image_no_background, trace_mask, sregion, trace_positions, [-trace_params["height"] / 3, trace_params["height"] / 3]; trace_pos_deg=extractor.trace_pos_deg)

    # Starting trace profile
    trace_profile = compute_vertical_trace_profile(trace_image, trace_mask, sregion, trace_positions, background=background, oversample=extractor.oversample_profile, aperture=[-ceil(trace_params["height"] / 2), ceil(trace_params["height"] / 2)])

    # Extract Aperture
    if isnothing(extractor.extract_aperture)
        extract_aperture = get_vertical_extract_aperture(trace_profile)
    else
        extract_aperture = copy(extractor.extract_aperture)
    end

    # Initial opt spectrum
    spec1d, spec1derr = optimal_extraction(trace_image, trace_mask, trace_positions, trace_profile, extract_aperture, background, background_err, read_noise, 1)

    # Main loop
    for i=1:extractor.n_iterations

        println(" Iteration $i")

        if extractor.remove_background
            trace_image_no_background = trace_image .- background'
        else
            trace_image_no_background = trace_image
        end
        
        trace_positions = Extract.compute_trace_positions_centroids(trace_image_no_background, trace_mask, sregion, trace_positions, extract_aperture; trace_pos_deg=extractor.trace_pos_deg)

        # Update trace profile with new positions and mask
        trace_profile = compute_vertical_trace_profile(trace_image, trace_mask, sregion, trace_positions, background=background, oversample=extractor.oversample_profile, aperture=[-ceil(trace_params["height"] / 2), ceil(trace_params["height"] / 2)])

        # Extract Aperture
        if isnothing(extractor.extract_aperture)
            extract_aperture = get_vertical_extract_aperture(trace_profile)
        else
            extract_aperture = copy(extractor.extract_aperture)
        end

        # Background
        if extractor.remove_background
            background, background_err = Extract.compute_background_1d(trace_image, trace_mask, trace_positions, extract_aperture, smooth_width=extractor.background_smooth_width)
        end

        # Optimal extraction
        spec1d, spec1derr = optimal_extraction(trace_image, trace_mask, trace_positions, trace_profile, extract_aperture, background, background_err, read_noise, 5)

        # Re-map pixels and flag in the 2d image.
        if i < extractor.n_iterations

            # 2d model
            spec1d_smooth = maths.median_filter1d(spec1d, 3)
            model2d_smooth = compute_model2d(extractor, trace_image, trace_mask, spec1d_smooth, trace_profile, trace_positions, extract_aperture, background)
            model2d_smooth = maths.median_filter2d(model2d_smooth, 3)

            # Flag
            n_bad_current = sum(trace_mask)
            Extract.flag_pixels2d!(trace_image, trace_mask, model2d_smooth, extractor.badpix_σ)
            n_bad_new = sum(trace_mask)
            
            # Break if nothing new is flagged but force 3 iterations
            if n_bad_current == n_bad_new && i > 1
                break
            end
        end
    end

    # 1d badpix mask
    spec1dmask = ones(nx)
    bad = findall(.~isfinite.(spec1d) .|| (spec1d .<= 0) .|| .~isfinite.(spec1derr) .|| (spec1derr .<= 0))
    spec1d[bad] .= NaN
    spec1derr[bad] .= NaN
    spec1dmask[bad] .= 0
    
    return (;spec1d=spec1d, spec1derr=spec1derr, spec1dmask=spec1dmask, trace_profile=trace_profile, trace_positions=trace_positions)
end


function optimal_extraction(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, trace_positions::Polynomials.Polynomial, trace_profile, extract_aperture, background=nothing, background_err=nothing, read_noise=0, n_iterations=1)

    # Image dims
    ny, nx = size(trace_image)
    
    # Helper array
    yarr = [1:ny;]

    # Outputs
    spec1d = fill(NaN, nx)
    spec1derr = fill(NaN, nx)

    # Tpy
    tpx, tpy = trace_profile.t, trace_profile.u

    # Use a consistent normalization factor to ensure P*f reproduces the data (up to noise)
    PA = DataInterpolations.integral(trace_profile, minimum(tpx), maximum(tpx))

    # Loop over iterations
    for i=1:n_iterations

        # Loop over cols
        for x=1:nx

            # Shift Trace Profile
            ymid = trace_positions(x)
            ybottom = ymid + extract_aperture[1]
            ytop = ymid + extract_aperture[2]

            P = maths.cspline_interp(tpx .+ ymid, tpy, yarr)
            
            # Determine which pixels to use from the aperture
            window = findall((yarr .>= ybottom) .&& (yarr .<= ytop))
            if length(window) < 1
                continue
            end

            # Profile
            P = P[window]
            if any(.~isfinite.(P))
                continue
            end
            P ./= PA
            #P ./= nansum(P)

            # Data
            if !isnothing(background)
                S = @views trace_image[window, x] .- background[x]
            else
                S = trace_image[window, x]
            end

            # Mask
            M = trace_mask[window, x]

            # Fix negative flux
            bad = findall(S .< 0)
            M[bad] .= 0
            S[bad] .= NaN

            # Check if column is worth extracting
            if sum(M) <= 1
                continue
            end

            # Variance
            if !isnothing(background)
                if i == 1
                    v = read_noise^2 .+ S .+ background[x] .+ background_err[x]^2
                else
                    v = read_noise^2 .+ spec1d[x] .* P .+ background[x] .+ background_err[x]^2
                end
            else
                if i == 1
                    v = read_noise^2 .+ S
                else
                    v = read_noise^2 .+ spec1d[x] .* P
                end
            end

            # Weights
            w = P.^2 .* M ./ v
            bad = findall(.~isfinite.(w))
            w[bad] .= 0
            w ./= sum(w)

            # Least squares
            #f = nansum(w .* S .* P) / nansum(w .* P.^2) # = nansum(P.^2 .* M .* S .* P ./ v) / nansum(P.^2 .* M .* P.^2)
            #ferr = sqrt(nansum(M .* P) / nansum(M .* P.^2 ./ v))

            # Horne
            f = nansum(M .* P .* S ./ v) / nansum(M .* P.^2 ./ v)
            ferr = sqrt(nansum(M .* P) / nansum(M .* P.^2 ./ v))
            #ferr = sqrt(nansum(w .* (S .- f .* P).^2))

            # Store
            spec1d[x] = f
            spec1derr[x] = ferr

        end
    end

    return spec1d, spec1derr
end

function optimal_extraction2(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, trace_positions::Polynomials.Polynomial, trace_profile, extract_aperture, background=nothing, background_err=nothing, read_noise=0, n_iterations=1)

    # Image dims
    ny, nx = size(trace_image)
    
    # Helper array
    yarr = [1:ny;]

    # Outputs
    spec1d = fill(NaN, nx)
    spec1derr = fill(NaN, nx)

    # Tpy
    tpx, tpy = trace_profile.t, trace_profile.u

    # Use a consistent normalization factor
    PA = DataInterpolations.integral(trace_profile, minimum(tpx), maximum(tpx))

    # Loop over iterations
    for i=1:n_iterations

        # Loop over cols
        for x=1:nx

            # Shift Trace Profile
            ymid = trace_positions(x)
            ybottom = ymid + extract_aperture[1]
            ytop = ymid + extract_aperture[2]

            P = maths.cspline_interp(tpx .+ ymid, tpy, yarr)
            
            # Determine which pixels to use from the aperture
            window = findall((yarr .>= ybottom) .&& (yarr .<= ytop))
            if length(window) < 1
                continue
            end

            # Profile
            P = P[window]
            if any(.~isfinite.(P))
                continue
            end
            #P ./= PA
            P ./= nansum(P)

            # Data
            if !isnothing(background)
                S = @views trace_image[window, x] .- background[x]
            else
                S = trace_image[window, x]
            end

            # Mask
            M = trace_mask[window, x]

            # Fix negative flux
            bad = findall(S .< 0)
            M[bad] .= 0
            S[bad] .= NaN

            # Check if column is worth extracting
            if sum(M) <= 1
                continue
            end

            # Variance
            if !isnothing(background)
                if i == 1
                    v = read_noise^2 .+ S .+ background[x] .+ background_err[x]^2
                else
                    v = read_noise^2 .+ spec1d[x] .* P .+ background[x] .+ background_err[x]^2
                end
            else
                if i == 1
                    v = read_noise^2 .+ S
                else
                    v = read_noise^2 .+ spec1d[x] .* P
                end
            end

            # Weights
            w = P.^2 .* M ./ v
            bad = findall(.~isfinite.(w))
            w[bad] .= 0
            w ./= sum(w)

            # Least squares
            #f = nansum(w .* S .* P) / nansum(w .* P.^2) # = nansum(P.^2 .* M .* S .* P ./ v) / nansum(P.^2 .* M .* P.^2)
            #ferr = sqrt(nansum(M .* P) / nansum(M .* P.^2 ./ v))

            # Horne
            f = nansum(M .* P .* S ./ v) / nansum(M .* P.^2 ./ v)
            ferr = sqrt(nansum(M .* P) / nansum(M .* P.^2 ./ v))
            #ferr = sqrt(nansum(w .* (S .- f .* P).^2))

            # Store
            spec1d[x] = f
            spec1derr[x] = ferr

        end
    end

    return spec1d, spec1derr
end

function compute_model2d(extractor::OptimalExtractor, trace_image::AbstractMatrix, trace_mask::AbstractMatrix, spec1d, trace_profile, trace_positions, extract_aperture, background=nothing)

    # Dims
    ny, nx = size(trace_image)

    # Helpful arr
    yarr = [1:ny;]

    # Initialize model
    model2d = fill(NaN, (ny, nx))

    # Trace profile at zero point
    tpx, tpy = trace_profile.t, trace_profile.u

    PA = DataInterpolations.integral(trace_profile, minimum(tpx), maximum(tpx))

    # Loop over cols
    for x=1:nx

        # Shift Trace Profile
        ymid = trace_positions(x)
        P = maths.cspline_interp(tpx .+ ymid, tpy, yarr)
        P ./= PA

        # Model
        model2d[:, x] .= P .* spec1d[x]
        if extractor.remove_background
            model2d[:, x] .+= background[x]
        end
    end

    # Return
    return model2d
end

function compute_vertical_trace_profile(trace_image, trace_mask, sregion, trace_positions; aperture, spec1d=nothing, background=nothing, oversample=1)
        
    # Image dims
    ny, nx = size(trace_image)
    
    # Helpful arrays
    yarr = [1:ny;]

    # Smooth
    trace_image_smooth = maths.median_filter2d(trace_image, 3)
    
    # Create a fiducial high resolution grid centered at zero
    yarr_hr0 = [Int(floor(-ny / 2)):(1 / oversample):Int(ceil(ny / 2));]
    trace_image_rect_norm = fill(NaN, (length(yarr_hr0), nx))

    # 1d spec info (smooth)
    if isnothing(spec1d)
        trace_image_smooth, _ = Extract.fix_bad_pixels_interp(trace_image_smooth, sregion.pixmin, sregion.pixmax, trace_positions + aperture[1], trace_positions + aperture[2])
        spec1d = NaNStatistics.nansum(trace_image_smooth, dim=1)
    end
    spec1d_smooth = maths.median_filter1d(spec1d, 3)
    spec1d_smooth ./= maths.weighted_median(spec1d_smooth, p=0.98)
    
    # Rectify
    for x=1:nx
        good = @views findall(isfinite.(trace_image[:, x]) .&& (trace_mask[:, x] .== 1))
        if length(good) >= 3 && spec1d_smooth[x] > 0.2
            ymid = trace_positions(x)
            if !isnothing(background)
                col_hr_shifted = @views maths.lin_interp(yarr .- ymid, trace_image_smooth[:, x] .- background[x], yarr_hr0)
            else
                col_hr_shifted = @views maths.lin_interp(yarr .- ymid, trace_image_smooth[:, x], yarr_hr0)
            end
            bad = findall(col_hr_shifted .< 0)
            col_hr_shifted[bad] .= NaN
            trace_image_rect_norm[:, x] .= col_hr_shifted ./ spec1d[x]
        end
    end
    
    # Compute trace profile
    trace_profile_median = nanmedian(trace_image_rect_norm, dim=2)

    # Compute cubic spline for profile and ignore edge vals
    good = findall(isfinite.(trace_profile_median))
    tpx = @views yarr_hr0[good[2:end-1]]
    tpy = @view trace_profile_median[good[2:end-1]]
    trace_profile = DataInterpolations.CubicSpline(tpy, tpx)

    # # Center profile at zero
    # prec = 1000
    # tpxhr = [tpx[1]:(1/prec):tpx[end];]
    # tpyhr = trace_profile(tpxhr)
    # mid = tpx[argmax(tpy)]
    # consider = findall((tpxhr .> mid - 3*oversample) & (tpxhr < mid + 3*self.oversample))[0]
    # trace_max_pos = tpxhr[consider[NaNargmax(tpyhr[consider])]]
    # trace_profile = scipy.interpolate.CubicSpline(trace_profile.x - trace_max_pos,
    #                                                         trace_profile(trace_profile.x), extrapolate=False)


    # # Final profile
    # tpx, tpy = trace_profile.x, trace_profile(trace_profile.x)
    # trace_profile = scipy.interpolate.CubicSpline(tpx, tpy ./ NaNmax(tpy), extrapolate=False)

    # Return
    return trace_profile
end


function get_vertical_extract_aperture(trace_profile)
    tpx, tpy = trace_profile.t, copy(trace_profile.u)
    tpy .-= nanminimum(tpy)
    tpy ./= nanmaximum(tpy)
    imax = maths.nanargmaximum(tpy)
    xleft = minimum(tpx)
    xright = maximum(tpx)
    for i=imax:-1:1
        if tpy[i] < 0.05
            xleft = tpx[i]
            break
        end
    end
    for i=imax:length(tpx)
        if tpy[i] < 0.05
            xright = tpx[i]
            break
        end
    end
    extract_aperture = [floor(xleft), ceil(xright)]
    return extract_aperture
end

end