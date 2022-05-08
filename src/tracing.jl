module Tracing

using NaNStatistics, Polynomials, Peaks, Statistics

using EchelleBase

export trace, gen_trace_image

function trace(data::SpecData2d, sregion::SpecRegion2d; trace_pos_deg=2, min_order_spacing, xleft=nothing, xright=nothing, n_slices=20, fiber=nothing)

    image = read_image(data)

    # dims
    ny, nx = size(image)

    # Number of orders
    n_orders = abs(sregion.ordertop - sregion.orderbottom) + 1

    # xleft and xright
    if isnothing(xleft)
        xleft = sregion.pixmin - 10
    end
    if isnothing(xright)
        xright = sregion.pixmax + 10
    end

    # Mask
    image = copy(image)
    mask!(image, sregion)

    # Smooth the image
    image_smooth = maths.median_filter2d(image, 3)

    # Slices
    xslices = Int.(round.(collect(range(xleft + 20, xright - 20, length=n_slices))))
    peaks = Vector{Any}(undef, length(xslices))
    heights = Vector{Any}(undef, length(xslices))
    good_slices = BitVector(undef, length(xslices))
    for i=1:n_slices
        x = xslices[i]
        s = @views reshape(nanmedian(image_smooth[:, x-5:x+5], dims=2), ny)
        good = @views findall(isfinite.(image_smooth[:, x-5:x+5]))
        ys = [coord.I[1] for coord ∈ good]
        yi, yf = minimum(ys), maximum(ys)
        background = maths.generalized_median_filter1d(s, width=3 * min_order_spacing, p=0.01)
        background[yi:yi+min_order_spacing] .= @views nanmedian(background[yi+min_order_spacing:yi+2*min_order_spacing])
        background[yf-min_order_spacing:yf] .= @views nanmedian(background[yf-2*min_order_spacing:yf-min_order_spacing])
        continuum = maths.generalized_median_filter1d(s .- background, width=3 * min_order_spacing, p=0.99)
        continuum[yi:yi+min_order_spacing] .= @views nanmedian(continuum[yi+min_order_spacing:yi+2*min_order_spacing])
        continuum[yf-min_order_spacing:yf] .= @views nanmedian(continuum[yf-2*min_order_spacing:yf-min_order_spacing])
        s .= (s .- background) ./ continuum
        good = findall(s .> 0.5)
        bad = findall(s .< 0.5)
        s[good] .= 1.0
        s[bad] .= 0.0

        # Peak finding
        _peaks, _heights = group_peaks(good; sep=min_order_spacing / 2)
        if length(_peaks) == n_orders
            peaks[i] = _peaks
            heights[i] = _heights
            good_slices[i] = true
        else
            peaks[i] = nothing
            heights[i] = nothing
            good_slices[i] = false
        end
    end

    # Fit
    ps = Polynomial[]
    heights_mean = Float64[]
    for i=1:n_orders
        xx = [xslices[k] for k=1:length(xslices) if good_slices[k]]
        yy = [peaks[k][i] for k=1:length(xslices) if good_slices[k]]
        _heights = [heights[k][i] for k=1:length(xslices) if good_slices[k]]
        pfit = Polynomials.fit(xx, yy, trace_pos_deg)
        push!(ps, pfit)
        push!(heights_mean, maths.weighted_median(_heights, p=0.1))
    end

    # Ensure the peaks are sorted from the bottom of the image to the top
    ys = [p.(nx/2) for p ∈ ps]
    ss = sortperm(ys)
    ps = [ps[ss[i]] for i=1:length(ss)]

    # Now build the orders list
    traces = []
    for i=1:length(ps)
        if sregion.orderbottom < sregion.ordertop
            order = sregion.orderbottom + i - 1
        else
            order = sregion.orderbottom - i + 1
        end
        if !isnothing(fiber)
            label = string(Int(order)) * "." * string(Int(fiber))
        else
            label = string(order)
        end
        push!(traces, Dict("fiber" => fiber, "height" => heights_mean[i], "poly" => ps[i], "order" => order, "label" => label))
    end

    # Sort according to orders
    orders = [d["order"] for d ∈ traces]
    ss = sortperm(orders)
    traces = [traces[ss[i]] for i=1:length(ss)]

    # Return
    return traces

end

function gen_trace_image(traces, ny, nx, sregion)

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
            
end