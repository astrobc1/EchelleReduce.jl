export calibrate_image!, gen_median_image, gen_median_dark, gen_median_flat


function calibrate_image!(
        image::Matrix{<:Real};
        bias_image::Union{Matrix{<:Real}, Nothing}=nothing,
        dark_image::Union{Matrix{<:Real}, Nothing}=nothing,
        flat_image::Union{Matrix{<:Real}, Nothing}=nothing,
        itime::Union{Real, Nothing}=nothing,
        dark_itime::Union{Real, Nothing}=nothing,
    )

    # Bias correction
    if !isnothing(bias_image)
        image .-= bias_image
    end
        
    # Dark correction
    if !isnothing(dark_image)
        if !isnothing(itime) && !isnothing(dark_itime)
            s = itime / dark_itime
        else
            s = 1
        end
        image .-= dark_image .* s
    end
        
    # Flat division
    if !isnothing(flat_image)
        image ./= flat_image
    end

    # Return the image
    return image
end


function gen_median_flat(
        flat_images::Vector{<:Matrix{<:Real}};
        bias_image::Union{Matrix{<:Real}, Nothing}=nothing,
        dark_image::Union{Matrix{<:Real}, Nothing}=nothing,
        flat_itime::Union{Real, Nothing}=nothing,
        dark_itime::Union{Real, Nothing}=nothing,
        q::Union{Real, Nothing}=0.5
    )

    # Coadd
    flat_image = median_combine_images(flat_images)

    # Bias subtraction
    if !isnothing(bias_image)
        flat_image .-= bias_image
    end

    # Dark subtraction
    if !isnothing(dark_image)
        if !isnothing(flat_itime) && !isnothing(dark_itime)
            s = flat_itime / dark_itime
        else
            s = 1
        end
        flat_image .-= dark_image .* s
    end

    # Flag negative vals
    bad = findall(flat_image .<= 0)
    flat_image[bad] .= NaN

    # Normalize
    if !isnothing(q)
        flat_image ./= nanquantile(flat_image, q)
    end

    # Return
    return flat_image
end


function gen_median_dark(
        dark_images::Vector{<:Matrix{<:Real}};
        bias_image::Union{Matrix{<:Real}, Nothing}=nothing,
    )

    # Coadd
    dark_image = median_combine_images(dark_images)

    # Bias subtraction
    if !isnothing(bias_image)
        dark_image .-= bias_image
    end

    # Return
    return dark_image

end


function median_combine_images(images::Vector{<:Matrix{<:Real}}; ignore_negative::Bool=true)
    ny, nx = size(images[1])
    image_out = fill(NaN, ny, nx)
    for x=1:nx
        for y=1:ny
            z = [image[y, x] for image in images]
            bad = ignore_negative ? findall(z .< 0 .|| .~isfinite.(z)) : findall(.~isfinite.(z))
            z[bad] .= NaN
            image_out[y, x] = nanmedian(z)
        end
    end
    return image_out
end