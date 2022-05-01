using Polynomials
using PyPlot
try
    plt.style.use("../gadfly_stylesheet.mplstyle")
catch
end

using EchelleBase
using EchelleReduce
using DataInterpolations

export extract_image, extract_trace, plot_extracted_spectrum

function extract_image(extractor::SpectralExtractor, data::SpecData2d, data_image::Matrix, sregion::SpecRegion2d, traces::Vector; badpix_mask=nothing, read_noise=0)
    
    # Store reduced data
    reduced_data = []

    # Loop over different traces
    for i=1:length(traces)
        trace_params = traces[i]
        println("[$(data)] Extracting Trace $(trace_params["label"])")
        #try
            ti = time()
            result = extract_trace(extractor, data_image, sregion, trace_params, badpix_mask=badpix_mask, read_noise=read_noise)
            push!(reduced_data, result)
            println("[$(data)] Extracted Trace $(trace_params["label"]) in $(round((time() - ti)/ 60, sigdigits=3)) min")

        #catch
        #    @warn "Warning! Could not extract trace $(trace["label"]) for $(data)"
        #    push!(reduced_data, nothing)
        #end
    end

    return reduced_data
end

function plot_extracted_spectrum(data::SpecData2d, reduced_data::Vector, sregion::SpecRegion2d, fname::String, traces::Vector)

    n_traces = length(traces)
    nx = length(reduced_data[1].spec1d)
    
    # The number of x pixels
    xarr = [1:nx;]
    
    # Plot settings
    plot_width = 30
    plot_height = 20
    dpi = 250
    n_rows = 4
    n_cols = Int(ceil(n_traces / n_rows))
    if n_cols > n_rows
        n_rows, n_cols = n_cols, n_rows
    end
    
    # Create a figure
    pygui(false)
    plt.style.use((@__DIR__)[1:end-3] * "gadfly_stylesheet.mplstyle")
    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi, squeeze=false)
    
    # For each subplot, plot all traces
    for row=1:n_rows
        for col=1:n_cols
            
            # The order index
            trace_index = n_cols * (row - 1) + col
            if trace_index > n_traces
                continue
            end

            trace_params = traces[trace_index]
            
            # Views
            spec1d = reduced_data[trace_index].spec1d
            spec1dmask = reduced_data[trace_index].spec1dmask
            
            # Good pixels
            good = findall(spec1dmask .== 1)
            if length(good) == 0
                continue
            end
            
            # Plot the extracted spectrum
            axarr[row, col].plot(xarr, spec1d, lw=1)

            # Title
            axarr[row, col].set_title("$(trace_params["label"])", fontsize=18)
            
            # Axis labels
            axarr[row, col].set_xlabel("X Pixels", fontsize=18)
            axarr[row, col].set_ylabel("Counts", fontsize=18)
            axarr[row, col].tick_params(labelsize=16)
        end
    end

    plt.suptitle("$(data)", fontsize=18)
    
    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(fname)
    plt.close()

end

function extract_trace end




compute_model2d(extractor::SpectralExtractor, args...) = nothing