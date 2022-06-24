
using EchelleBase

using PyCall
using Infiltrator
using Polynomials
using NaNStatistics


"""
    estimate_snr(trace_image::AbstractMatrix)
Crude method to estimate the mean S/N.
"""
function estimate_snr(trace_image::AbstractMatrix)
    spec1d = nansum(trace_image, dim=1)
    spec1d_smooth = nansum(maths.median_filter2d(trace_image, 5), dim=1)
    med_val = maths.weighted_median(spec1d_smooth, p=0.98)
    spec1d ./= med_val
    spec1d_smooth ./= med_val
    res_norm = spec1d .- spec1d_smooth
    snr = 1 / nanstd(res_norm)
    return snr
end


"""
    refine_trace_window(image::AbstractMatrix, badpix_mask::AbstractMatrix, sregion::SpecRegion2d, trace_positions_estimate::Polynomial; trace_pos_poly_deg=nothing, window::Vector, n_iterations=3)
Iteratively refines the initial trace window using the centroids of each column.
"""
function refine_trace_window(image::AbstractMatrix, badpix_mask::AbstractMatrix, sregion::SpecRegion2d, trace_positions_estimate::Polynomial; trace_pos_poly_deg=nothing, window::Vector, n_iterations=3)

    # The image dimensions
    ny, nx = size(image)

    # Initial positions from polynomial
    ycen = trace_positions_estimate.(1:nx)
    trace_positions = deepcopy(trace_positions_estimate)

    if isnothing(trace_pos_poly_deg)
        trace_pos_poly_deg = degree(trace_positions)
    end

    for i=1:n_iterations

        # Copy the image
        image_cp = copy(image)

        # Mask image according to sregion
        mask!(image_cp, sregion)

        # Copy
        trace_image = copy(image_cp)
        trace_mask = copy(badpix_mask)

        # Mask according to positions
        for x=1:nx
            y_low = Int(floor(ycen[x] + window[1]))
            y_high = Int(ceil(ycen[x] + window[2]))
            if y_low >= 1 && y_low <= ny
                trace_image[1:y_low, x] .= NaN
            else
                trace_image[:, x] .= NaN
            end
            if y_high >= 1 && y_high <= ny
                trace_image[y_high+1:end, x] .= NaN
            else
                trace_image[:, x] .= NaN
            end
        end

        # Sync
        bad = findall(.~isfinite.(trace_image) .|| (trace_mask .== 0))
        trace_image[bad] .= NaN
        trace_mask[bad] .= 0

        # Compute centroids and polynomial fit
        trace_positions = compute_trace_positions_centroids(trace_image, trace_mask, sregion, trace_positions, window; trace_pos_poly_deg=trace_pos_poly_deg)

    end

    # Return
    return trace_positions

end

"""
    compute_trace_positions_centroids(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, sregion::SpecRegion2d, trace_positions::Polynomial, aperture; trace_pos_poly_deg=nothing)
Compute the trace positions by fitting a polynomial to the column centroids.
"""
function compute_trace_positions_centroids(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, sregion::SpecRegion2d, trace_positions::Polynomial, aperture; trace_pos_poly_deg=nothing)

    # The image dimensions
    ny, nx = size(trace_image)
    
    # Helpful arrays
    yarr = 1:ny

    # Fix nans
    trace_image, trace_mask = fix_bad_pixels_interp(trace_image, sregion.pixmin, sregion.pixmax, trace_positions + aperture[1], trace_positions + aperture[2])

    # Smoothed 1d spectrum
    spec1d = maths.median_filter1d(nansum(maths.median_filter2d(trace_image, 3), dim=1), 3)
    med_val = maths.weighted_median(spec1d, p=0.98)
    spec1d ./= med_val

    # Y centroids
    ycen = fill(NaN, nx)

    # Loop over columns
    for x=sregion.pixmin:sregion.pixmax
        ymid = trace_positions(x)
        if !isnothing(aperture)
            use = @views findall((trace_mask[:, x] .== 1) .&& isfinite.(trace_image[:, x]) .&& (yarr .>= ymid + aperture[1]) .&& (yarr .<= ymid + aperture[2]))
        else
            use = @views findall((trace_mask[:, x] .== 1) .&& isfinite.(trace_image[:, x]))
        end
        if length(use) <= 2 || spec1d[x] < 0.2
            continue
        end

        # Centroid
        ycen[x] = @views maths.weighted_mean(use, trace_image[use, x])
    end

    # Flag
    ycen_smooth = maths.median_filter1d(ycen, 3)
    bad = findall(abs.(ycen - ycen_smooth) .> 1)
    ycen[bad] .= NaN

    # Fit
    good = findall(isfinite.(ycen))
    xarr = [1:nx;]
    trace_positions = @views Polynomials.fit(xarr[good], ycen[good], trace_pos_poly_deg)

    # Return
    return trace_positions
end

"""
    compute_background_1d(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, trace_positions::Polynomial, extract_aperture; smooth_width=nothing)
Estimates the background signal by considering regions of low flux according to the provided `extract_aperture`.
"""
function compute_background_1d(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, trace_positions::Polynomial, extract_aperture; smooth_width=nothing)
    ny, nx = size(trace_image)
    xarr = [1:nx;]
    yarr = [1:ny;]
    background = fill(NaN, nx)
    for x=1:nx
        ymid = trace_positions(x)
        ybottom = ymid + extract_aperture[1]
        ytop = ymid + extract_aperture[2]
        use = @views findall(((yarr .< ybottom) .|| (yarr .> ytop)) .&& isfinite.(trace_image[:, x]) .&& (trace_mask[:, x] .== 1))
        if length(use) > 0
            background[x] = @views nanmedian(trace_image[use, x])
        else
            background[x] = @views nanminimum(trace_image[:, x])
        end
    end
    if !isnothing(smooth_width) && smooth_width > 3
        background = maths.poly_filter(xarr, background, width=smooth_width, deg=2)
    end
    bad = findall(background .< 0)
    background[bad] .= 0
    background_err = sqrt.(background)
    return background, background_err
end

"""
    flag_pixels2d!(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, model2d::AbstractMatrix, nσ::Real)
Flags bad pixels in the 2d image by looking at outliers in the residuals.
"""
function flag_pixels2d!(trace_image::AbstractMatrix, trace_mask::AbstractMatrix, model2d::AbstractMatrix, nσ::Real)

    # Smooth the 2d image to normalize residuals
    trace_image_smooth = maths.median_filter2d(trace_image, 3)

    # Normalized residuals
    norm_res = (trace_image .- model2d) ./ sqrt.(trace_image_smooth)

    # Flag
    use = findall(.~(norm_res .== 0) .&& (trace_mask .== 1) .&& isfinite.(norm_res))
    σ = @views maths.robust_σ(norm_res[use], nσ=100)
    bad = findall(abs.(norm_res) .> nσ * σ)
    trace_mask[bad] .= 0
    trace_image[bad] .= NaN
end

"""
    fix_bad_pixels_interp(trace_image::AbstractMatrix, xmin, xmax, poly_bottom::Polynomial, poly_top::Polynomial)
Utility function to fix bad pixels in an image with linear interpolation.
"""
function fix_bad_pixels_interp(trace_image::AbstractMatrix, xmin, xmax, poly_bottom::Polynomial, poly_top::Polynomial)
    ny, nx = size(trace_image)
    good = findall(isfinite.(trace_image))
    bad = findall(.~isfinite.(trace_image))
    trace_image_out = copy(trace_image)
    trace_image_out[bad] .= 0
    xx = repeat(1:nx, outer=(1, ny))'
    yy = repeat(1:ny, outer=(1, nx))
    x1 = xx[good]
    y1 = yy[good]
    scipy_interp = pyimport("scipy.interpolate")
    trace_image_out .= scipy_interp.griddata((x1, y1), trace_image[good], (xx, yy), method="linear")
    yarr = 1:ny
    for x=xmin:xmax
        ybottom = poly_bottom(x)
        ytop = poly_top(x)
        bad = findall(yarr .< ybottom .|| yarr .> ytop)
        trace_image_out[bad, x] .= NaN
    end
    trace_image_out[:, 1:xmin-1] .= NaN
    trace_image_out[:, xmax:end] .= NaN
    trace_mask_out = ones(size(trace_image_out))
    bad = findall(.~isfinite.(trace_image_out))
    trace_mask_out[bad] .= NaN
    return trace_image_out, trace_mask_out
end
