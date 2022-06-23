export reduce, initialize_data, create_output_dirs, gen_master_calib_images, get_master_flat, get_master_dark, get_traces, extract

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

# Extraction
function extract end