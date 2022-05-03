using EchelleBase
using EchelleReduce

mutable struct SPExtractor <: SpectralExtractor
    trace_pos_deg::Int
    oversample::Int
    n_iterations::Int
    chunk_width::Int
    badpix_σ::Int
    σpoly::Polynomial
    extract_aperture::Int
    qpoly::Polynomial
    θpoly::Polynomial
end

function get_psf_matrix(trace_image, xa, trace_positions, σpoly, qpoly, θpoly, aperture)

    # Image dims
    ny, nx = size(trace_image)

    # Aperture size (half)
    aperture_size_half = Int(ceil((aperture[2] - aperture[1]) / 2))

    # Initialize A
    n_apertures = length(xa)
    A = zeros(ny, nx, n_apertures)

    # Helpful arrays
    xd = [1:nx;]
    yd = [1:ny;]
    ya = trace_positions.(xa) .+ 1

    # PSF Parameters
    σ = σpoly.(xa)
    q = qpoly.(xa)
    θ = θpoly.(xa)

    # Loops!
    for m=1:n_apertures
        for i=1:nx
            for j=1:ny

                # (x, y) of the center of the aperture, relative to dims of image
                xkc = xa[m]
                ykc = ya[m]

                # Coordinates
                # (x, y) of the psf, relative to dims of image
                xl = xd[i]
                yl = yd[j]

                # Diffs
                Δx = xl - xkc
                Δy = yl - ykc
                if abs(Δx) > aperture_size_half || abs(Δy) > aperture_size_half
                    continue
                end

                # Tilted Coordinates relative to center of aperture
                xp = Δx * sin(θ[m]) - Δy * cos(θ[m])
                yp = Δx * cos(θ[m]) + Δy * sin(θ[m])

                # Compute PSF
                A[j, i, m] = exp(-0.5 * ((xp / σ[m])^2 + (yp / (q[m] * σ[m]))^2))
            end
        end
    end


    # Normalize each aperture
    for m=1:n_apertures

        # Normalize each aperture
        s = sum(A[:, :, m])
        if s != 0
            A[:, :, m] ./= s
        end
    end

    return A
end
    

function extract_SP2d(A, S, Ninv)
    
    # Convert to sparse
    A = SparseMatrixCSC(reshape(A, (size(A)[1] * size(A)[2], size(A)[3])))
    S = SparseMatrixCSC(reshape(S, (length(S), 1)))
    Ninv = SparseMatrixCSC(Ninv)
    
    # Compute helpful vars
    Cinv = transpose(A) * Ninv * A
    result = svd(collect(Cinv))
    U = result.U
    Vt = result.Vt
    S = result.S
    Cpsuedo = transpose(Vt) * diagm(1 ./ S) * transpose(U)
    Cpsuedo = SparseMatrixCSC(Cpsuedo)

    # Initial flux
    flux = Cpsuedo * (transpose(A) * Ninv * S)
    flux = collect(flux)
        
    # Compute reconvolution matrix
    Cinv = SparseMatrixCSC(Cinv)
    f, Wt = eigvals(Cinv, minimum(size(Cinv)) - 2)
    F = diagm(f)
    F .= abs.(F)
    
    # Faster to do dense than sparse (at least in my one test session)
    Wt = real.(Wt)
    WtDhW = Wt * sqrt.(F) * transpose(Wt)
    ss = sum(WtDhW, dims=2)[:]
    Sinv = inv(diagm(ss))
    R = Sinv * WtDhW
    
    # Reconvolve
    fluxtilde = R * flux

    # Convert to final arrays
    flux = reshape(flux, nx)
    fluxt = reshape(flux, nx)

    return flux, fluxt, R
end

function compute_model2d(data, trace_image, trace_mask, trace_params, spec1d, trace_positions, σpoly, qpoly, θpoly, extract_aperture)

    # Dims
    ny, nx = size(trace_image)

    # Number of chunks
    chunks = generate_chunks(trace_image, trace_mask)

    # Copy input spectrum and fix nans (inside bounds)
    f = copy(spec1d)
    xarr = [1:nx;]
    good = findall(isfinite.(f))
    bad = np.where(.~isfinite.(f))
    f[bad] .= @views maths.lin_interp(xarr[good], f[good], xarr[bad])
    bad = findall(.~isfinite.(f))
    f[bad] = 0

    # Stitch points
    model2d = fill(NaN, (ny, nx, length(chunks))

    # Loop over chunks
    for i=1:length(chunks)
        xxi, xxf = chunks[i][1], chunks[i][2]
        nnnx = xxf - xxi + 1
        good = @views findall(trace_mask[:, xxi:xxf])
        goody = [coord.I[1] for coord ∈ good]
        yyi, yyf = minimum(goody) maximum(goody)
        nnny = yyf - yyi + 1
        S = @view trace_image[yyi:yyf, xxi:xxf]
        xarr_aperture = [xxi:xxf;]
        A = get_psf_matrix(S, xarr_aperture, trace_positions, σpoly, qpoly, θpoly, aperture)
        Aflat = reshape(A, (nnny*nnnx, nnnx))
        model2d[yyi:yyf+1, xxi:xxf+1, i] = @views reshape(Aflat * f[xxi:xxf+1], (nnny, nnnx))
        model2d[:, xxi:xxi+20, i] .= NaN
        model2d[:, xxf-20:xxf+1, i] .= NaN
    end

    # Average chunks
    model2d = nanmean(model2d, dims=3)[:, :]

    return model2d
end


function generate_chunks(trace_image, trace_mask)
    good = findall(trace_mask .== 1)
    goody = [coord.I[1] for coord ∈ good]
    goodx = [coord.I[2] for coord ∈ good]
    xi, xf = minimum(goodx), maximum(goodx)
    nnx = xf - xi + 1
    yi, yf = minimum(goody), maximum(goody)
    nny = yf - yi + 1
    chunk_width = min(chunk_width, 200)
    chunks = []
    push!(chunks, (xi, xi + chunk_width))
    for i=2:Int(2 * ceil(nnx / chunk_width))
        vi = chunks[i-1][1] - Int(floor(chunk_width / 2))
        vf = min(vi + chunk_width, xf)
        push!(chunks, (vi, vf))
        if vf == xf
            break
        end
    end
    return chunks
end

function bin_spec1d(spec1dhr, oversample)
    nx = Int(length(spec1dhr) / oversample)
    return nansum(reshape(spec1dhr, (nx, oversample)), dims=2)[:]
end

function Extract.extract_trace(extractor::SPExtractor, trace_image::Matrix, trace_mask::Matrix, trace_positions::Polynomial, extract_aperture; read_noise=0)
    
    # Copy input
    trace_image_cp = copy(trace_image)
    trace_mask_cp = copy(trace_mask)

    # Fix bad pixels
    fix_bad_pixels_interp!(trace_image_cp, trace_mask_cp)

    # Dims
    ny, nx = size(trace_image_cp)

    # Flag negative pixels
    bad = findall(trace_image_cp .< 0)
    trace_image_cp[bad] .= NaN
    trace_mask_cp[bad] .= 0

    # Now set all nans to zero
    bad = findall(.~isfinite.(trace_image_cp) .|| (trace_mask_cp .== 0))
    trace_image_cp[bad] .= 0
    trace_mask_cp[bad] .= 0

    # Chunks
    chunks = generate_chunks(trace_image, trace_mask_cp, chunk_width)
    n_chunks = length(chunks)

    # Outputs (averaged over chunks before returning)
    spec1d = fill(NaN, (nx, n_chunks)
    spec1dt = fill(NaN, (nx, n_chunks)
    R = fill(NaN, (nx, nx, n_chunks)

    # Loop over and extract chunks
    for i=1:n_chunks

        # X pixel bounds for this chunk
        xi, xf = chunks[i][1], chunks[i][2]
        nnx = xf - xi + 1

        # Y pixel bounds  for this chunk
        good = @views findall(trace_mask_cp[:, xi:xf] .== 1)
        goody = [coord.I[1] for coord ∈ good]
        yi, yf = minimum(goody), maxmimum(goody)
        nny = yf - yi + 1

        # Crop image and mask to this chunk
        S = @view trace_image_cp[yi:yf, xi:xf]
        M = @view trace_mask_cp[yi:yf, xi:xf]

        # X centroid of each aperture in index units
        xarr_aperture = [(-0.5 + 0.5 / oversample):(1 / oversample):(nnx - 1 + 0.5 - 0.5 / oversample);] .+ 1

        # Generate Aperture tensor for this chunk
        A = get_psf_matrix(S, xarr_aperture, trace_positions, extractor.σpoly, extractor.qpoly, extractor.θpoly, extract_aperture)

        # Prep inputs for sparse extraction
        Aflat = reshape(A, (nny*nnx, length(xarr_aperture)))
        Sflat = collect(Iterators.flatten(S))
        Ninv = 1 ./ (S + read_noise**2)
        bad = findall(.~isfinite.(Ninv))
        Ninv[bad] .= 0
        Wbig = diagm(Iterators.flatten(Ninv))

        # Call sparse extraction
        syhr, sythr, _Rhr = extract_SP2d(Aflat, Sflat, Wbig)

        # Bin back to detector grid
        sy = bin_spec1d(syhr)
        syt = bin_spec1d(sythr)

        # Mask edge errors
        distrust_width = Int(maximum(abs.(extract_aperture)) * 2)
        sy[1:distrust_width] .= NaN
        sy[-distrust_width:end] .= NaN
        syt[1:distrust_width] .= NaN
        syt[-distrust_width:end] .= NaN
        _Rhr[1:distrust_width, 1:distrust_width] .= NaN
        _Rhr[-distrust_width:end, -distrust_width:end] .= NaN

        # Store results
        spec1d[xi:xf+1, i] .= sy
        spec1dt[xi:xf+1, i] .= syt
        spec1d[xi:xf+1, i] .= sy
        spec1dt[xi:xf+1, i] .= syt
        R[xi:xf+1, xi:xf+1, i] .= _Rhr
    end

    # Final trim of edges
    spec1d[1:xi+distrust_width, 1] .= NaN
    spec1d[xf-distrust_width:end, end-1] .= NaN
    spec1dt[1:xi+distrust_width, 1] .= NaN
    spec1dt[xf-distrust_width:end, end-1] .= NaN
    R[:, 1:xi+distrust_width, 1] .= NaN
    R[:, xf-distrust_width:end, end-1] .= NaN

    # Correct negatives and zeros in reconvolved spectrum
    bad = findall(spec1dt .<= 0)
    spec1dt[bad] .= NaN

    # Average each chunk
    spec1d = nanmean(spec1d, dims=2)[:]
    spec1d_unc = sqrt.(spec1d)
    spec1dt = np.nanmean(spec1dt, dims=2)[:]
    spec1dt_unc = sqrt.(spec1dt)

    # For R, make sure each psf is normalized to sum=1
    RR = nanmean(R, dims=2)[:, :]
    for i=1:nx
        RR[i, :] ./= @views nansum(RR[i, :])
    end
    goody, goodx = np.where(np.isfinite(RR))
    good = findall(isfinite.(RR))
    goody = [coord.I[1] for coord ∈ good]
    xi, xf = minimum(goodx), maximum(goodx)
    for x=1:nx
        try
            RR[x, 1:xi+20] .= 0
            RR[x, xf - 20:] .= 0
        catch
            nothing
        end
    end

    bad = findall(RR .< 0)
    RR[bad] .= 0
    for x=1:nx
        RR[x, :] ./= @views nansum(RR[x, :])
    end

    RR[1:xi-20, :] .= 0
    RR[xf - 20:end, :] .= 0

    bad = findall((RR .<= 0) .|| .~isfinite.(RR))
    RR[bad] = 0

    bad = findall(.~.isfinite.(spec1dt) .|| (spec1dt .<= 0))
    spec1dt[bad] .= NaN

    # Return
    return spec1d, spec1d_unc, spec1dt, spec1dt_unc, RR

end

function Extract.extract_trace(extractor::SPExtractor, image::Matrix, sregion::SpecRegion1d, trace_params::Dict; badpix_mask=nothing, read_noise=0.0)

    # Copy image
    image = copy(image)

    # Full dims
    ny, nx = size(image)

    # Initiate mask
    if isnothing(badpix_mask)
        badpix_mask = ones(ny, nx)
    else
        badpix_mask = copy(badpix_mask)
    end

    # Refine initial window
    trace_positions = Extract.refine_initial_trace_window(image, badpix_mask, sregion, trace_params, n_iterations=3)

    # Mask image based on trace aperture
    trace_image = copy(image)
    trace_mask = copy(badpix_mask)
    for x=1:nx
        ymid = trace_positions(x) + 1
        y_low = Int(floor(ymid - trace_params["height"] / 2))
        y_high = Int(ceil(ymid + trace_params["height"] / 2))
        if y_low > 1 && y_low < ny
            trace_image[1:y_low-1, x] .= NaN
        else
            trace_image[:, x] .= NaN
        end
        if y_high > 1 && y_high < ny
            trace_image[y_high+1:end, x] .= NaN
        else
            trace_image[:, x] .= NaN
        end
    end

    # Sync
    bad = findall(.~isfinite.(trace_image) .|| (trace_mask == 0))
    trace_image[bad] .= NaN
    trace_mask[bad] .= 0

    # Crop in the y direction
    good = findall(isfinite.(trace_image))
    goody = [coord.I[1] for coord ∈ good]
    yi, yf = minimum(goody), maximum(goody)
    trace_image = @view trace_image[yi:yf, :]
    trace_mask = @view trace_mask[yi:yf, :]
    ny, nx = size(trace_image)
    trace_positions -= yi

    # Flag obvious bad pixels again
    trace_image_smooth = maths.median_filter2d(trace_image, 5)
    peak = maths.weighted_median(trace_image_smooth, p=0.99)
    bad = findall((trace_image .< 0) .|| (trace_image .> 50 * peak))
    trace_image[bad] .= NaN
    trace_mask[bad] .= 0

    # Initial spectrum
    spec1d, spec1derr = extract_trace(extractor, trace_image, trace_mask, trace_positions, read_noise=0)

    # Main loop
    for i=1:extractor.n_iterations

        println(" Iteration $i")

        trace_positions = Extract.compute_trace_positions_centroids(trace_image_no_background, trace_mask, sregion, trace_positions, extract_aperture; trace_pos_deg=extractor.trace_pos_deg)

        # Extract Aperture
        if isnothing(extractor.extract_aperture)
            extract_aperture = get_vertical_extract_aperture(trace_profile)
        else
            extract_aperture = copy(extractor.extract_aperture)
        end

        # Optimal extraction
        spec1d, spec1derr = extract_trace(extractor, trace_image, trace_mask, trace_positions, trace_profile, extract_aperture, background, background_err, read_noise, 5)

        # Re-map pixels and flag in the 2d image.
        if i < extractor.n_iterations

            # 2d model
            spec1d_smooth = maths.median_filter1d(spec1d, 3)
            model2d_smooth = compute_model2d(extractor, trace_image, trace_mask, spec1d_smooth, trace_profile, trace_positions, extract_aperture, background)
            model2d_smooth = maths.median_filter2d(model2d_smooth, 3)

            # Flag
            n_bad_current = sum(trace_mask)
            Extract.flag_pixels2d!(trace_image, trace_mask, model2d_smooth, extractor.badpix_σ)
            n_bad_new = sum(trace_mask)
            
            # Break if nothing new is flagged but force 3 iterations
            if n_bad_current == n_bad_new && i > 1
                break
            end
        end
    end

    # 1d badpix mask
    spec1dmask = ones(nx)
    bad = findall(.~isfinite.(spec1d) .|| (spec1d .<= 0) .|| .~isfinite.(spec1derr) .|| (spec1derr .<= 0))
    spec1d[bad] .= NaN
    spec1derr[bad] .= NaN
    spec1dmask[bad] .= 0
    
    return (;spec1d=spec1d, spec1derr=spec1derr, spec1dmask=spec1dmask, reconv_matrix)

end 