export trace_boxcar

function trace_boxcar(
        image::Matrix{<:Real}, labels::Vector{String};
        xrange::Vector{Int}, yrange::Vector{Int},
        x0::Union{Int, Nothing}=nothing,
        min_spacing::Real,
        deg::Int,
        width_bounds::Vector{<:Real},
        σ_bounds::Vector{<:Real},
        fit_window::Union{Int, Nothing}=nothing,
        flux_smooth_width::Union{Int, Nothing}=nothing,
        peak_thresh::Real=0.5
    )
    
    # Copy the image since we will modify it
    image = copy(image)

    # Image dims
    ny, nx = size(image)

    # Number of orders to trace
    n_traces = length(labels)

    # Mask L/R of image
    image[:, .~(xrange[1] .<= axes(image, 2) .<= xrange[2])] .= NaN

    # Default x starting slice
    if isnothing(x0)
        x0 = Int(round((xrange[1] + xrange[2]) / 2))
    end

    # Smooth the image
    image_smooth = quantile_filter(image, window=(3, 3))

    # Initial ycol
    xi, xf = max(xrange[1], x0 - 5), min(xrange[2], x0 + 5)
    zcol_start = nanmedian(view(image_smooth, :, xi:xf), dim=2)

    # Mask
    zcol_start[.~(yrange[1] .<= eachindex(zcol_start) .<= yrange[2])] .= NaN

    # Replace nans with zeros
    bad = filter(y -> yrange[1] .<= y .<= yrange[2], findall(.~isfinite.(zcol_start)))
    zcol_start[bad] .= 0

    # Get initial ycens
    ycens_start, _ = get_starting_peaks(zcol_start; min_spacing, flux_smooth_width, thresh=peak_thresh)

    # Confirm correct number found
    @assert length(ycens_start) == length(labels) "Number of peaks found ($(length(ycens_start)) != Number of labels ($length(labels))"

    # Fit initial slice
    ycens = [fill(NaN, nx) for _ in 1:n_traces]
    ycen_errs = [fill(NaN, nx) for _ in 1:n_traces]
    widths = [fill(NaN, nx) for _ in 1:n_traces]
    width_errs = [fill(NaN, nx) for _ in 1:n_traces]
    σs = [fill(NaN, nx) for _ in 1:n_traces]
    σ_errs = [fill(NaN, nx) for _ in 1:n_traces]
    for i in eachindex(labels)
        μ_bounds = [-fit_window / 3, fit_window / 3]
        μ0 = ycens_start[i]
        result = fit_trace_position_gaussboxcar(zcol_start; μ0, μ_bounds, width_bounds, σ_bounds, fit_window)
        if !isnothing(result)
            ycens[i][x0] = result.μ
            ycen_errs[i][x0] = result.μ_err
            widths[i][x0] = result.width
            width_errs[i][x0] = result.width_err
            σs[i][x0] = result.σ
            σ_errs[i][x0] = result.σ_err
        end
    end

    # Work left and right to compute centroids for each order every pixels
    for x=(x0-1):-1:xrange[1]
        zcol = nanmedian(quantile_filter(view(image, :, max(x-2, 1):min(x+2, nx)), window=(3, 3)), dim=2)
        for i in eachindex(labels)
            μ_bounds = [-fit_window / 3, fit_window / 3]
            if isfinite(ycens[i][x+1])
                μ0 = ycens[i][x+1]
            else
                good = findall(isfinite.(ycens[i]))
                if length(good) == 0
                    continue
                end
                k = good[minimum(abs.(good .- x))]
                μ0 = ycens[i][k]
            end
            result = fit_trace_position_gaussboxcar(zcol; μ0, μ_bounds, width_bounds, σ_bounds, fit_window)
            if !isnothing(result)
                ycens[i][x] = result.μ
                ycen_errs[i][x] = result.μ_err
                widths[i][x] = result.width
                width_errs[i][x] = result.width_err
                σs[i][x] = result.σ
                σ_errs[i][x] = result.σ_err
            end
        end
    end
    for x=(x0+1):1:xrange[2]
        zcol = nanmedian(quantile_filter(view(image, :, max(x-2, 1):min(x+2, nx)), window=(3, 3)), dim=2)
        for i in eachindex(labels)
            μ_bounds = [-fit_window / 3, fit_window / 3]
            if isfinite(ycens[i][x-1])
                μ0 = ycens[i][x-1]
            else
                good = findall(isfinite.(ycens[i]))
                if length(good) == 0
                    continue
                end
                k = good[minimum(abs.(good .- x))]
                μ0 = ycens[i][k]
            end
            result = fit_trace_position_gaussboxcar(zcol; μ0, μ_bounds, width_bounds, σ_bounds, fit_window)
            if !isnothing(result)
                ycens[i][x] = result.μ
                ycen_errs[i][x] = result.μ_err
                widths[i][x] = result.width
                width_errs[i][x] = result.width_err
                σs[i][x] = result.σ
                σ_errs[i][x] = result.σ_err
            end
        end
    end

    # Fit with a polynomial
    polys = Vector{ArnoldiFit}(undef, n_traces)
    for i in eachindex(labels)
        polys[i], _ = polyfit1d(1:nx, ycens[i]; deg)
    end

    # New trace widths from fits
    heights = [nanmedian(widths[i] .- 5 .* σs[i]) for i in eachindex(labels)]

    # Collect auxiliary data from fits
    traces = OrderedDict(labels[i] => (;
        label=labels[i],
        yc=polys[i].(1:nx),
        yrange=[-heights[i] / 2, heights[i] / 2],
        widths=widths[i],
        width_errs=width_errs[i],
        σs=σs[i],
        σ_errs=σ_errs[i]) for i ∈ eachindex(labels)
    )

    # Return
    return traces

end


function gaussboxcar_model(x, p)
    xx = x .- p[2]
    y = @. p[1] * (erf((xx + p[3] / 2) / p[4]) - erf((xx - p[3] / 2) / p[4]))^2 / 4 + p[5]
    return y
end


function fit_trace_position_gaussboxcar(
    z::Vector{Float64};
    μ0::Real, μ_bounds::Vector{<:Real}, width_bounds::Vector{<:Real}, σ_bounds::Vector{<:Real},
    fit_window::Union{Nothing, Real}=nothing,
)

    # Fit window
    if isnothing(fit_window)
        fit_window = Int(round(2.5 * width_bounds[2]))
    end

    # Window
    n = length(z)
    yy = max(floor(μ0 - fit_window / 2), 1):min(ceil(μ0 + fit_window / 2), n)
    zz = z[Int.(yy)]

    # Good data
    good = findall(isfinite.(zz))
    if length(good) < 5
        return nothing
    end
    yy, zz = yy[good], zz[good]

    # Initial parameters
    # Amp, mean, width, sigma, b
    zb = nanminimum(zz)
    za = nanmaximum(zz)
    amp0 = za - zb
    p0 = [amp0, μ0, nanmean(width_bounds), nanmean(σ_bounds), zb]
    lb = [amp0 * 0.5, μ0 + μ_bounds[1], width_bounds[1], σ_bounds[1], zb - zb / 2]
    ub = [amp0 * 1.5, μ0 + μ_bounds[2], width_bounds[2], σ_bounds[2], zb + zb / 2]

    # Try to fit
    out = nothing
    try

        # Fit
        result = LsqFit.curve_fit(gaussboxcar_model, yy, zz, p0, lower=lb, upper=ub, autodiff=:forwarddiff)
        rms = sqrt(nansum(result.resid.^2) / length(result.resid))

        # Errors
        pbest = result.param
        pbest_err = LsqFit.standard_errors(result)
        
        # Result
        amp = pbest[1]
        amp_err = pbest_err[1]
        μ = pbest[2]
        μ_err = pbest_err[2]
        width = pbest[3]
        width_err = pbest_err[3]
        σ = pbest[4]
        σ_err = pbest_err[4]
        background = pbest[5]
        background_err = pbest_err[5]
        out = (;amp, amp_err, μ, μ_err, width, width_err, σ, σ_err, background, background_err, rms)
    catch
        @warn "LM fit failed for μ0=$μ0"
    end

    # Return
    return out

end