

export plot_reduced_spectrum, plot_tracepos_image

function plot_reduced_spectrum(
        reduced_data::OrderedDict, filename::String;
        plot_width::Int=30, plot_height::Int=20, dpi::Int=250, n_rows::Int=4,
        itime::Union{Real, Nothing}=nothing, gain::Union{Real, Nothing}=nothing,
    )

    # The number of traces to plot
    trace_labels = collect(keys(reduced_data))
    n_traces = length(trace_labels)
    
    # Number of cols
    n_cols = Int(ceil(n_traces / n_rows))
    if n_cols > n_rows
        n_rows, n_cols = n_cols, n_rows
    end
    
    # Create a figure
    pygui(false)
    _, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi, squeeze=false)

    # Loop over reduced traces
    for i=1:n_traces

        # Label
        trace_label = trace_labels[i]

        # Skip is reduction failed
        if isnothing(reduced_data[trace_label])
            continue
        end

        # Copy 1D spectrum
        spec1d = copy(reduced_data[trace_label].spec)
        yl = "e-"
        if !isnothing(gain) && !isnothing(itime)
            spec1d ./= (gain .* itime)
            yl = "ADU/s"
        end

        # 1D -> 2D index
        row, col = Int(floor((i - 1) / n_cols)) + 1, ((i - 1) % n_cols) + 1

        # Plot
        axarr[row, col].plot(1:length(spec1d), spec1d, lw=1, c=COLORS_GADFLY_HEX[1])

        # Labels
        axarr[row, col].set_title(trace_label, fontsize=18)
        axarr[row, col].set_xlabel("X Pixels", fontsize=18)
        axarr[row, col].set_ylabel(yl, fontsize=18)
        axarr[row, col].tick_params(labelsize=16)
    end

    # Figure title
    plt.suptitle(basename(filename), fontsize=18)
    
    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(filename)
    plt.close()

end

function plot_tracepos_image(
        image::Matrix{<:Real},
        ycs::Vector{<:Vector{<:Real}},
        filename::String;
        qmax::Real=0.98
    )
    ny, nx = size(image)
    xarr = 1:nx
    pygui(false)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    imshow(image, extent=(0.5, nx+.5, ny+.5, .5), vmin=0, vmax=nanquantile(image, qmax))
    for yc in ycs
        plot(xarr, yc, c="red")
    end
    ylim(0, ny+1)
    plt.title(basename(filename))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
end


# Similar to plot_tracepos_image but for debugging
function plot_tracepos_image_debug(
        image::Matrix{<:Real},
        ycs::Union{Vector{<:Real}, Vector{<:Vector{<:Real}}};
        q::Real=0.95
    )
    ny, nx = size(image)
    xarr = 1:nx
    fig, ax = plt.subplots()
    imshow(image, extent=(0.5, nx+.5, ny+.5, .5), vmin=0, vmax=nanquantile(image, q))
    if ycs isa Vector{<:Real}
        ycs = [ycs]
    end
    for yc in ycs
        plot(xarr, yc, c="red")
    end
    ylim(0, ny)
    return fig, ax
end

