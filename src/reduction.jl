using EchelleBase

export reduce, initialize_data, create_output_dirs, gen_master_calib_images, get_master_flat, get_master_dark, get_traces, get_trace_spacing, get_trace_height, get_specregion2d, get_extract_orders, get_read_noise, extract

# Main function
function reduce end
"""
    reduce(recipe::ReduceRecipe)
Perform reduction steps according to the recipe. The default steps are:
`create_output_dirs(recipe)`
`data = initialize_data(recipe)`
`gen_master_calib_images(recipe, data)`
`traces = get_traces(recipe, data)`
`extract(recipe, data, traces, sregions)`
"""
function reduce(recipe::ReduceRecipe)
    create_output_dirs(recipe)
    data = initialize_data(recipe)
    gen_master_calib_images(recipe, data)
    traces = get_traces(recipe, data)
    extract(recipe, data, traces, sregions)
end

"""
    initialize_data
Method intended to initialize parse the input directory and categorize the data. Must be implemented.
"""
function initialize_data end

"""
    create_output_dirs
Method intended to create any output directories. Must be implemented.
"""
function create_output_dirs end

# Calibration
"""
    gen_master_calib_images
Method intended to generate all calibration frames. Must be implemented.
"""
function gen_master_calib_images end

"""
    get_master_flat
Method intended to get the appropriate master flat frame for calibration. Must be implemented.
"""
function get_master_flat end

"""
    get_master_dark
Method intended to get the appropriate master dark frame for calibration. Must be implemented.
"""
function get_master_dark end

# Tracing
"""
    get_traces
Method intended to get the appropriate vector of trace params. Must be implemented.
"""
function get_traces end

"""
    get_specregion2d
Returns the minimum trace spacing.
"""
function get_trace_spacing end

"""
    get_specregion2d
Returns the maximum trace height.
"""
function get_trace_height end

"""
    get_specregion2d
Returns the appropriate SpecRegion2d
"""
function get_specregion2d end

# Extraction
"""
    get_extract_orders
Gets the orders to be extracted. Must be implemented.
"""
function get_extract_orders end

"""
    get_read_noise(itime::Real, dark_current::Real=0, read_noise::Real=0)
    get_read_noise(data::SpecData, dark_current::Real=0, read_noise::Real=0)
Compute the effective read noise in units of photoelectrons.
"""
function get_read_noise end
function get_read_noise(itime::Real, dark_current::Real=0, read_noise::Real=0)
    return itime * dark_current + read_noise
end
function get_read_noise(data::SpecData, dark_current::Real=0, read_noise::Real=0)
    return get_read_noise(parse_itime(data), dark_current, read_noise)
end

"""
    extract
Method intended to perform all necessary extraction steps. Must be implemented.
"""
function extract end