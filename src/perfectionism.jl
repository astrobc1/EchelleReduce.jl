module SPExtraction

using EchelleBase
using EchelleReduce

using Polynomials
using NaNStatistics
using Infiltrator
using LinearAlgebra
using SparseArrays
using Arpack

export SPExtractor

mutable struct SPExtractor <: SpectralExtractor
    trace_pos_deg::Int
    oversample::Int
    n_iterations::Int
    chunk_width::Int
    badpix_σ::Int
    extract_aperture::Int
    σpoly::Polynomial
    qpoly::Polynomial
    θpoly::Polynomial
end

function SPExtractor(;trace_pos_deg, oversample=1, n_iterations=20, chunk_width=100, badpix_σ=5, extract_aperture, σpoly, qpoly, θpoly)
    return SPExtractor(trace_pos_deg, oversample, n_iterations, chunk_width, badpix_σ, extract_aperture, σpoly, qpoly, θpoly)
end

function get_psf_tensor(trace_image, xa, ya, xd, yd, trace_positions, σpoly, qpoly, θpoly, aperture)

    # Image dims
    ny, nx = size(trace_image)

    # Aperture size (half)
    aperture_size_half = Int(ceil(aperture / 2))

    # Initialize A
    n_apertures = length(xa)
    A = zeros(ny, nx, n_apertures)

    # PSF Parameters
    σ = σpoly.(xa)
    q = qpoly.(xa)
    θ = θpoly.(xa)

    # Loops!
    for m=1:n_apertures

        # Centroid of the aperture
        xkc = xa[m]
        ykc = trace_positions(xkc)

        for i=1:nx
            for j=1:ny

                # Coordinates of the psf for this aperture
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
    A = SparseMatrixCSC(A)
    S = SparseMatrixCSC(reshape(S, (length(S), 1)))
    Ninv = SparseMatrixCSC(Ninv)
    
    # Compute helpful vars
    Cinv = transpose(A) * Ninv * A
    result = svd(collect(Cinv))
    U = result.U
    Vt = result.Vt
    ss = result.S
    Cpsuedo = transpose(Vt) * diagm(1 ./ ss) * transpose(U)
    Cpsuedo = SparseMatrixCSC(Cpsuedo)

    # Initial flux
    flux = Cpsuedo * (transpose(A) * Ninv * S)
    flux = collect(flux)
        
    # Compute reconvolution matrix
    Cinv = SparseMatrixCSC(Cinv)
    f, Wt = eigs(Cinv, nev=minimum(size(Cinv)) - 2)
    F = diagm(f)
    F .= abs.(F)
    
    # Faster to do dense than sparse (at least in my one test session)
    Wt = real.(Wt)
    WtDhW = Wt * sqrt.(F) * transpose(Wt)
    ss = sum(WtDhW, dims=2)[:]
    Sinv = inv(diagm(ss))
    R = Sinv * WtDhW
    
    # Reconvolve
    fluxt = R * flux

    # Convert to final arrays
    flux = reshape(real.(flux), length(flux))
    fluxt = reshape(real.(fluxt), length(fluxt))

    return flux, fluxt, R
end

function Extract.compute_model2d(extractor::SPExtractor, trace_image::AbstractMatrix, trace_mask::AbstractMatrix, spec1d::AbstractVector, trace_positions::Polynomial)

    # Dims
    ny, nx = size(trace_image)

    # Number of chunks
    chunks = generate_chunks(trace_image, trace_mask, extractor.chunk_width)

    # Copy input spectrum and fix nans (inside bounds)
    f = copy(spec1d)
    xarr = [1:nx;]
    good = findall(isfinite.(f))
    bad = findall(.~isfinite.(f))
    f[bad] .= @views maths.lin_interp(xarr[good], f[good], xarr[bad])
    bad = findall(.~isfinite.(f))
    f[bad] .= 0

    # Stitch points
    model2d = fill(NaN, (ny, nx, length(chunks)))
    distrust_width = Int(extractor.extract_aperture * 2)

    # Loop over chunks
    for i=1:length(chunks)
        xi, xf = chunks[i][1], chunks[i][2]
        nnx = xf - xi + 1
        good = @views findall(trace_mask[:, xi:xf] .== 1)
        goody = [coord.I[1] for coord ∈ good]
        yi, yf = minimum(goody), maximum(goody)
        nny = yf - yi + 1
        S = @view trace_image[yi:yf, xi:xf]
        xarr_aperture = [(xi - 0.5 + 0.5 / extractor.oversample):(1 / extractor.oversample):(xf + 0.5 - 0.5 / extractor.oversample);]
        yarr_aperture = [(yi - 0.5 + 0.5 / extractor.oversample):(1 / extractor.oversample):(yf + 0.5 - 0.5 / extractor.oversample);]
        A = get_psf_tensor(S, xarr_aperture, yarr_aperture, xi:xf, yi:yf, trace_positions, extractor.σpoly, extractor.qpoly, extractor.θpoly, extractor.extract_aperture)
        Aflat = reshape(A, (nny*nnx, nnx))
        model2d[yi:yf, xi:xf, i] = @views reshape(Aflat * f[xi:xf], (nny, nnx))
        model2d[:, xi:xi+distrust_width, i] .= NaN
        model2d[:, xf-distrust_width:xf, i] .= NaN
    end

    # Average chunks
    model2d = nanmean(model2d, dims=3)[:, :]

    return model2d
end


function generate_chunks(trace_image, trace_mask, chunk_width)
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
        vi = chunks[i-1][2] - Int(floor(chunk_width / 2))
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

function Extract.extract_trace(extractor::SPExtractor, trace_image::AbstractMatrix, trace_mask::AbstractMatrix, sregion::SpecRegion2d, trace_positions::Polynomial; read_noise=0)
    
    # Copy input
    trace_image_cp = copy(trace_image)
    trace_mask_cp = copy(trace_mask)

    # Fix bad pixels
    trace_image_cp, trace_mask_cp = Extract.fix_bad_pixels_interp(trace_image, sregion.pixmin, sregion.pixmax, trace_positions - extractor.extract_aperture - 1, trace_positions + extractor.extract_aperture + 1)

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
    chunks = generate_chunks(trace_image, trace_mask_cp, extractor.chunk_width)
    n_chunks = length(chunks)

    # Outputs (averaged over chunks before returning)
    spec1d = fill(NaN, (nx, n_chunks))
    spec1dt = fill(NaN, (nx, n_chunks))
    R = fill(NaN, (nx, nx, n_chunks))

    distrust_width = Int(extractor.extract_aperture * 2)

    # Loop over and extract chunks
    for i=1:n_chunks

        # X pixel bounds for this chunk
        xi, xf = chunks[i][1], chunks[i][2]
        nnx = xf - xi + 1

        # Y pixel bounds  for this chunk
        good = @views findall(trace_mask_cp[:, xi:xf] .== 1)
        goody = [coord.I[1] for coord ∈ good]
        yi, yf = minimum(goody), maximum(goody)
        nny = yf - yi + 1

        # Crop image and mask to this chunk
        S = @view trace_image_cp[yi:yf, xi:xf]
        M = @view trace_mask_cp[yi:yf, xi:xf]

        # Aperture arrays
        xarr_aperture = [(xi - 0.5 + 0.5 / extractor.oversample):(1 / extractor.oversample):(xf + 0.5 - 0.5 / extractor.oversample);]
        yarr_aperture = [(yi - 0.5 + 0.5 / extractor.oversample):(1 / extractor.oversample):(yf + 0.5 - 0.5 / extractor.oversample);]

        # Generate Aperture tensor for this chunk
        A = get_psf_tensor(S, xarr_aperture, yarr_aperture, xi:xf, yi:yf, trace_positions, extractor.σpoly, extractor.qpoly, extractor.θpoly, extractor.extract_aperture)

        # Prep inputs for sparse extraction
        Aflat = reshape(A, (nny*nnx, length(xarr_aperture)))
        Sflat = collect(Iterators.flatten(S))
        Ninv = 1 ./ (S .+ read_noise^2)
        bad = findall(.~isfinite.(Ninv))
        Ninv[bad] .= 0
        Wbig = diagm(collect(Iterators.flatten(Ninv)))

        # Call sparse extraction
        syhr, sythr, _Rhr = extract_SP2d(Aflat, Sflat, Wbig)

        # Bin back to detector grid
        sy = bin_spec1d(syhr, extractor.oversample)
        syt = bin_spec1d(sythr, extractor.oversample)

        # Mask edge errors
        sy[1:distrust_width] .= NaN
        sy[end-distrust_width:end] .= NaN
        syt[1:distrust_width] .= NaN
        syt[end-distrust_width:end] .= NaN
        _Rhr[1:distrust_width, 1:distrust_width] .= NaN
        _Rhr[end-distrust_width:end, end-distrust_width:end] .= NaN

        # Store results
        spec1d[xi:xf, i] .= sy
        spec1dt[xi:xf, i] .= syt
        spec1d[xi:xf, i] .= sy
        spec1dt[xi:xf, i] .= syt
        #R[xi:xf, xi:xf, i] .= _Rhr
    end

    @infiltrate

    # Final trim of edges
    spec1d[1:xi+distrust_width, 1] .= NaN
    spec1d[xf-distrust_width:end, end-1] .= NaN
    spec1dt[1:xi+distrust_width, 1] .= NaN
    spec1dt[xf-distrust_width:end, end-1] .= NaN
    #R[:, 1:xi+distrust_width, 1] .= NaN
    #R[:, xf-distrust_width:end, end-1] .= NaN

    # Correct negatives and zeros in reconvolved spectrum
    bad = findall(spec1dt .<= 1E-5)
    spec1dt[bad] .= NaN

    # Average each chunk
    spec1d = nanmean(spec1d, dims=2)[:]
    #spec1d_unc = sqrt.(spec1d)
    spec1d_unc = ones(length(spec1d))
    spec1dt = nanmean(spec1dt, dims=2)[:]
    spec1dt_unc = ones(length(spec1dt))
    #spec1dt_unc = sqrt.(spec1dt)

    # For R, make sure each psf is normalized to sum=1
    #RR = nanmean(R, dims=3)[:, :]
    #for i=1:nx
    #    RR[i, :] ./= @views nansum(RR[i, :])
    #end
    #good = findall(isfinite.(RR))
    goody = [coord.I[1] for coord ∈ good]
    goodx = [coord.I[2] for coord ∈ good]
    xi, xf = minimum(goodx), maximum(goodx)
    # for x=1:nx
    #     try
    #         RR[x, 1:xi+distrust_width] .= 0
    #         RR[x, xf - distrust_width:end] .= 0
    #     catch
    #         nothing
    #     end
    # end

    #bad = findall(RR .< 0)
    #RR[bad] .= 0
    #for x=1:nx
    #    RR[x, :] ./= @views nansum(RR[x, :])
    #end

    #RR[1:xi-distrust_width, :] .= 0
    #RR[xf - distrust_width:end, :] .= 0

    #bad = findall((RR .< 0) .|| .~isfinite.(RR))
    #RR[bad] .= 0

    bad = findall(.~isfinite.(spec1dt) .|| (spec1dt .<= 0))
    spec1dt[bad] .= NaN

    # Return
    return spec1d, spec1d_unc, spec1dt, spec1dt_unc, RR

end

function Extract.extract_trace(extractor::SPExtractor, image::Matrix, sregion::SpecRegion2d, trace_params::Dict; badpix_mask=nothing, read_noise=0.0)

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

    # Initial trace positions
    aperture = [-extractor.extract_aperture, extractor.extract_aperture]
    trace_positions = Extract.compute_trace_positions_centroids(trace_image, trace_mask, sregion, trace_positions, aperture; trace_pos_deg=extractor.trace_pos_deg)

    # Initial spectrum
    spec1d, spec1derr, spec1dt, spec1dterr, RR = extract_trace(extractor, trace_image, trace_mask, sregion, trace_positions, read_noise=0)

    # Main loop
    for i=1:extractor.n_iterations

        println(" Iteration $i")

        trace_positions = Extract.compute_trace_positions_centroids(trace_image, trace_mask, sregion, trace_positions, aperture; trace_pos_deg=extractor.trace_pos_deg)

        # SP extraction
        spec1d, spec1derr, spec1dt, spec1dterr, RR = extract_trace(extractor, trace_image, trace_mask, sregion, trace_positions, read_noise=0)

        # Re-map pixels and flag in the 2d image.
        if i < extractor.n_iterations

            # 2d model
            #spec1dt_smooth = maths.median_filter1d(spec1dt, 3)
            model2d = Extract.compute_model2d(extractor, trace_image, trace_mask, spec1dt, trace_positions)
            #model2d_smooth = maths.median_filter2d(model2d_smooth, 3)

            # Flag
            n_bad_current = sum(trace_mask)
            Extract.flag_pixels2d!(trace_image, trace_mask, model2d, extractor.badpix_σ)
            n_bad_new = sum(trace_mask)
            
            # Break if nothing new is flagged but force 3 iterations
            if n_bad_current == n_bad_new && i > 1
                break
            end
        end
    end

    # 1d badpix mask
    spec1dmask = ones(nx)
    bad = findall(.~isfinite.(spec1dt) .|| (spec1dt .<= 0) .|| .~isfinite.(spec1dterr) .|| (spec1dterr .<= 0))
    spec1dt[bad] .= NaN
    spec1dterr[bad] .= NaN
    spec1dmask[bad] .= 0
    
    return (;spec1d=spec1dt, spec1derr=spec1dterr, spec1dmask=spec1dmask)

end 

end