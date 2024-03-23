export LSQBivariateSpline

struct LSQBivariateSpline
    x::Any
    y::Any
    z::Any
    tx::Any
    ty::Any
    w::Any
    kx::Int
    ky::Int
    spl::PyObject
end


function LSQBivariateSpline(x, y, z, tx, ty; w=nothing, kx=3, ky=3)
    scipyinterp = pyimport("scipy.interpolate")
    spl = scipyinterp.LSQBivariateSpline(x, y, z, tx, ty; w, kx, ky)
    LSQBivariateSpline(x, y, z, tx, ty, w, kx, ky, spl)
end


function (spl::LSQBivariateSpline)(x, y)
    return spl.spl(x, y)
end


struct LSQBivariateSplineSerialization
    x::Any
    y::Any
    z::Any
    tx::Any
    ty::Any
    w::Any
    kx::Int
    ky::Int
end

JLD2.writeas(::Type{LSQBivariateSpline}) = LSQBivariateSplineSerialization

function Base.convert(::Type{LSQBivariateSplineSerialization}, spl::LSQBivariateSpline)
    LSQBivariateSplineSerialization(spl.x, spl.y, spl.z, spl.tx, spl.ty, spl.w, spl.kx, spl.ky)
end


function Base.convert(::Type{LSQBivariateSpline}, spl::LSQBivariateSplineSerialization) 
    LSQBivariateSpline(spl.x, spl.y, spl.z, spl.tx, spl.ty; spl.w, spl.kx, spl.ky)
end