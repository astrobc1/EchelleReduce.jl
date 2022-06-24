using EchelleBase

export reduce, initialize_data, create_output_dirs, gen_master_calib_images, get_master_flat, get_master_dark, get_traces, get_trace_spacing, get_trace_height, get_specregion2d, get_extract_orders, get_read_noise, extract

# Main function
function reduce end
function reduce(recipe::ReduceRecipe)
    create_output_dirs(recipe)
    data = initialize_data(recipe)
    gen_master_calib_images(recipe, data)
    traces = get_traces(recipe, data)
    extract(recipe, data, traces, sregions)
end

# Initialize data
function initialize_data end

# Create output dirs
function create_output_dirs end

# Calibration
function gen_master_calib_images end
function get_master_flat end
function get_master_dark end

# Tracing
function get_traces end
function get_trace_spacing end
function get_trace_height end
function get_specregion2d end

# Extraction
function get_extract_orders end
function get_read_noise end
function get_read_noise(itime, dark_current=0, read_noise=0)
    return itime * dark_current + read_noise
end
function get_read_noise(data::SpecData, dark_current=0, read_noise=0)
    return get_read_noise(parse_itime(data), dark_current, read_noise)
end
function extract end