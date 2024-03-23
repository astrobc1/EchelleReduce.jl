export repair_bad_pix2d

function repair_bad_pix2d(
        image::Matrix{<:Real}, yc::Vector{Float64};
        yrange::Vector{<:Real}, xrange::Vector{Int}
    )

    # Number of y pixels
    ny = size(image, 1)

    # Copy image (result)
    image_out = copy(image)

    # Helpfull yarr
    yarr = 1:ny

    # Loop over columns
    for x=xrange[1]:xrange[2]

        # Sort pixels in this column by distance from center of trace
        # We repair pixels in order by their distance from the trace centroid.
        ys = sortperm(abs.(Int.(round.(collect(yarr) .- yc[x]))))

        # Loop over bad pixels for this column
        for y in ys

            # Check if pixel is finite and within trace limits
            if !isfinite(image_out[y, x]) && (yc[x] + yrange[1]) < y < (yc[x] + yrange[2])

                # Get repaired val
                v = interp2d_badpix(image_out, y, x)

                # Ensure val is finite
                # If not, try NN
                if isfinite(v)
                    image_out[y, x] = v
                else
                    image_out[y, x] = repair_pixel_nearest_neighbors(image_out, y, x)
                end
            end
        end
    end

    # Ensure no negative pixels
    clamp!(image_out, 0, Inf)

    # Return
    return image_out
end


function repair_pixel_nearest_neighbors(image::Matrix{<:Real}, y::Int, x::Int)
    ny, nx = size(image)
    yi = max(y - 1, 1)
    yf = min(y + 1, ny)
    xi = max(x - 1, 1)
    xf = min(x + 1, nx)
    return nanmedian(view(image, yi:yf, xi:xf))
end

function interp2d_badpix(image::AbstractMatrix{T}, x, y)::T where {T<:Real}

    # Image dims
    nx, ny = size(image)
    
    # Get good pixels above and below
    k1x = get_good_neighbors_x_low(image, x, y)
    k2x = get_good_neighbors_x_high(image, x, y)
    k1y = get_good_neighbors_y_low(image, x, y)
    k2y = get_good_neighbors_y_high(image, x, y)
    if k1x < 1 || k2x < 1 || k1y < 1 || k2y < 1
        return NaN
    end
    
    Q11 = image[x, k1y]
    Q12 = image[x, k2y]
    Q21 = image[k1x, y]
    Q22 = image[k2x, y]

    f = (1 / ((k2x - k1x) * (k2y - k1y))) * [k2x - x x - k1x] * [[Q11, Q21] [Q12, Q22]] * [k2y - y, y - k1y]
    f = f[1]

    return f
end

function get_good_neighbors_x_low(image, x, y)
    t = @views image[:, y]
    for _x=x:-1:1
        if isfinite(t[_x])
            return _x
        end
    end
    return -1
end

function get_good_neighbors_x_high(image, x, y)
    nx = size(image, 1)
    t = @views image[:, y]
    for _x=x:nx
        if isfinite(t[_x])
            return _x
        end
    end
    return -1
end

function get_good_neighbors_y_low(image, x, y)
    t = @views image[x, :]
    for _y=y:-1:1
        if isfinite(t[_y])
            return _y
        end
    end
    return -1
end

function get_good_neighbors_y_high(image, x, y)
    ny = size(image, 2)
    t = @views image[x, :]
    for _y=y:ny
        if isfinite(t[_y])
            return _y
        end
    end
    return -1
end