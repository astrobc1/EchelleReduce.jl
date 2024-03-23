
export extract_trace, Extractor, ExtractionResult, save_auxiliary_data

abstract type Extractor end

function extract_trace(filename::String, image::Matrix{<:Real}, trace_label::String, extractor::Extractor)

    # Result for failure
    result = nothing

    # Try to extract
    try

        println("Extracting $(basename(filename)), trace $(trace_label) ...")

        # Timer
        ti = time()
        
        # Extract
        result = extract_trace(image, extractor)

        # Print duration
        println("Extracted $(basename(filename)), trace $(trace_label) in $(round((time() - ti) / 60, digits=3)) min")

    catch e
        
        @error "Could not extract $(basename(filename)), trace $(trace_label)" exception=(e, catch_backtrace())

    finally

       GC.gc()

    end

    # Return
    return result

end


function save_auxiliary_data(filename::String, reduced_data::OrderedDict{String, <:Any})
    auxiliary = OrderedDict{String, Any}()
    for key in keys(reduced_data)
        auxiliary[key] = OrderedDict{String, Any}()
        if !isnothing(reduced_data[key])
            for akey in keys(reduced_data[key].auxiliary)
                auxiliary[key][akey] = reduced_data[key].auxiliary[akey]
            end
        end
    end
    jldsave(filename; auxiliary)
end
