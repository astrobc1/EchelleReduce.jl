module PreCalib

using NaNStatistics
using EchelleBase

export gen_master_bias, gen_master_dark, gen_master_flat, gen_image_cube, pre_calibrate!, gen_master_coadded_image

function gen_image_cube(data::Vector)
    n_images = length(data)
    image0 = read_image(data[1])
    ny, nx = size(image0)
    image_cube = fill(NaN, (n_images, ny, nx))
    image_cube[1, :, :] .= image0
    if n_images > 1
        for i=2:n_images
            image_cube[i, :, :] .= read_image(data[i])
        end
    end
    return image_cube
end

function gen_master_bias(bias_data::Vector)
    bias_cube = gen_image_cube(bias_data)
    mbias = nanmedian(bias_cube, dim=1)
    return mbias
end


function gen_master_dark(master_dark::MasterCal2d; master_bias=nothing)

    # Generate a data cube
    darks_cube = gen_image_cube(master_dark.group)

    # Median crunch
    mdark = nanmedian(darks_cube, dim=1)

    # Bias subtraction
    if !isnothing(master_bias)
        mdark .-= master_bias
    end
    
    # Change negative pixels to zero
    bad = findall(mdark .<= 0)
    mdark[bad] .= 0

    # Return
    return mdark
end


function gen_master_flat(master_flat::MasterCal2d; master_bias=nothing, master_dark=nothing, p=0.5)
   
    # Generate a data cube
    flats_cube = gen_image_cube(master_flat.group)

    # Median crunch
    mflat = nanmedian(flats_cube, dim=1)

    # Dark and Bias subtraction
    if !isnothing(master_bias)
        mflat .-= master_dark
    end
    if !isnothing(master_bias)
        mflat .-= master_dark
    end

    # Normalize
    mflat ./= maths.weighted_median(mflat, p=p)
    
    # Flag obvious bad pixels
    bad = findall(mflat .<= 0)
    mflat[bad] .= NaN

    # Return
    return mflat
end

function pre_calibrate!(image::Matrix; master_bias=nothing, master_dark=nothing, master_flat=nothing)
    
    # Bias correction
    if !isnothing(master_bias)
        mbias = read_image(master_bias)
        image .-= mbias
    end
        
    # Dark correction
    if !isnothing(master_dark)
        mdark = read_image(master_dark)
        image .-= mdark
    end
        
    # Flat division
    if !isnothing(master_flat)
        mflat = read_image(master_flat)
        image ./= mflat
    end
end

function gen_master_coadded_image(master_frame::SpecData2d)
    
    # Generate a data cube
    image_cube = gen_image_cube(master_frame.group)

    # Median crunch
    master_image = nanmedian(image_cube, dim=1)
    
    # Flag obvious bad pixels
    bad = findall(master_image .< 0)
    master_image[bad] .= NaN
        
    # Return
    return master_image
end

#function compute_inter_order_background(image, orders_list)
#end

end