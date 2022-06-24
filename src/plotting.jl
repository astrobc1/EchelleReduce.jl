using PyPlot
using EchelleBase
using Infiltrator

export plot_extracted_spectrum

function plot_extracted_spectrum(recipe::ReduceRecipe, data::SpecData2d, reduced_data::Vector, fname::String, traces::Vector)

    n_traces = length(traces)
    
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
            
            if isnothing(reduced_data[trace_index])
                continue
            end

            nx = length(reduced_data[1].spec1d)
    
            # The number of x pixels
            xarr = [1:nx;]
            
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
