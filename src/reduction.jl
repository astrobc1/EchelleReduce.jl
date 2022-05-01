export ReduceRecipe, initialize_data, create_output_dirs, gen_master_calib_images, trace, extract, reduce

abstract type ReduceRecipe end

function reduce end
function initialize_data end
function create_output_dirs end
function gen_master_calib_images end
function trace end
function extract end