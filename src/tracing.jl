
abstract type Tracer end

struct TraceResult
    label::String
    yc::Vector{Float64}
    yrange::Vector{Float64}
    auxillary::Dict{String, Any}
end


function get_starting_peaks(zcol; min_spacing::Int, flux_smooth_width::Int=10 * min_spacing, thresh::Real=0.5)
    ny = length(zcol)
    if flux_smooth_width > ny
        flux_smooth_width = ny
    end
    if !isodd(flux_smooth_width)
        flux_smooth_width += 1
    end
    zcol_smooth = quantile_filter(zcol, window=3)
    background = quantile_filter(zcol_smooth, window=flux_smooth_width, q=0)
    background .= quantile_filter(background, window=5)
    zcol_nobg = zcol .- background
    zcol_nobg .-= nanminimum(zcol_nobg)
    continuum = quantile_filter(zcol_nobg, window=flux_smooth_width, q=1)
    continuum .= quantile_filter(continuum, window=5, q=0.5)
    zcol_norm = zcol_nobg ./ continuum
    good = findall(isfinite.(zcol_norm) .&& (zcol_norm .> thresh))
    ycens, heights = group_trace_peaks(good, min_spacing)
    ycens = Float64.(ycens)
    return ycens, heights
end


function group_trace_peaks(x::Vector, sep::Real)
    peak_centers = Float64[]
    peak_widths = Float64[]
    prev_i = 1
    for i=1:length(x) - 1
        if x[i+1] - x[i] > sep
            push!(peak_centers, (x[prev_i] + x[i]) / 2)
            push!(peak_widths, x[i] - x[prev_i])
            prev_i = i + 1
        end
    end
    push!(peak_centers, (x[prev_i] + x[end]) / 2)
    push!(peak_widths, x[end] - x[prev_i])
    return peak_centers, peak_widths
end
