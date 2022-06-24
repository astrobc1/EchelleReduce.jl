using EchelleBase
using EchelleReduce

export extract_image, extract_trace, gen_model2d

"""
    extract_image(extractor::SpectralExtractor, data::SpecData2d, data_image::AbstractMatrix, sregion::SpecRegion2d, traces::Vector; badpix_mask::Union{AbstractMatrix, Nothing}=nothing, read_noise::Real=0)
Primary method to extract all orders from an image.
"""
function extract_image(extractor::SpectralExtractor, data::SpecData2d, data_image::AbstractMatrix, sregion::SpecRegion2d, traces::Vector; badpix_mask::Union{AbstractMatrix, Nothing}=nothing, read_noise::Real=0)
    
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
        #    push!(reduced_data, nothing)
        #    @warn "Warning! Could not extract trace $(trace_params["label"]) for $(data)"
        #end
    end

    return reduced_data
end


"""
    extract_trace
Primary method to extract a single trace. Must be implemented.
"""
function extract_trace end

"""
    extract_trace
Primary method to generate a 2d model. Must be implemented.
"""
function gen_model2d end