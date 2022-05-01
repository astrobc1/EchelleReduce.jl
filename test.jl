push!(LOAD_PATH, "/Users/cale/Development/EchelleBase/src/")
push!(LOAD_PATH, "/Users/cale/Development/EchelleReduce/src/")
push!(LOAD_PATH, "/Users/cale/Development/EchelleSpectralModeling/src/")
push!(LOAD_PATH, "/Users/cale/Development/EchelleSpectrographs/src/")

using EchelleBase
using EchelleReduce
using EchelleSpectrographs.ishell
using Polynomials

# Basic info
data_input_path = "/Users/cale/Research/Spectrographs/iSHELL/Raw/TOI_461_20190911UT/"
#data_input_path = "/Users/cale/Research/Spectrographs/iSHELL/Raw/Vega_Test/"
base_flat_field_file = "/Users/cale/Research/Spectrographs/iSHELL/Reduced/ByNight/Vega_Test/calib/master_flat_20161016_imgs21-25.fits"
output_path = "/Users/cale/Research/Spectrographs/iSHELL/Reduced/ByNight/"
#extract_orders = [219, 220, 221]
extract_orders = [212:240;]

# Image region
sregion = SpecRegion2d(;pixmin=299, pixmax=1747, orderbottom=212, ordertop=240, poly_bottom=Polynomial([-116.36685525376339, 0.20359022197025314, -5.9597390213793886e-05]), poly_top=Polynomial([1858.343750000002, 0.1634374999999993, -5.078125000000014e-05]))

# Extractor
extractor = OptimalExtractor(;trace_pos_deg=4, oversample_profile=8, badpix_σ=4, remove_background=true, n_iterations=20)

# Create the recipe
recipe = iSHELLReduceRecipe(data_input_path=data_input_path, output_path=output_path,
                            sregion=sregion, do_dark=false, do_flat=true,
                            order_height=28, order_spacing=25,
                            base_flat_field_file=base_flat_field_file,
                            extractor=extractor, extract_orders=extract_orders, n_cores=1)

# Reduce the night
EchelleReduce.reduce(recipe)