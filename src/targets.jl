using Base.Order: Ordering
import Base.Order: lt

abstract type Target end

"""
For finding eigenvalues with the Largest Magnitude
"""
struct LM <: Target end

"""
For finding eigenvalues with the Largest Real part
"""
struct LR <: Target end

"""
For finding eigenvalues with the Smallest Real part
"""
struct SR <: Target end

"""
For finding eigenvalues with the Largest Imaginary part
"""
struct LI <: Target end

"""
For finding eigenvalues with the Smallest Imaginary part
"""
struct SI <: Target end

struct EigenvalueOrder{T<:Target,R<:RitzValues} <: Ordering
    ritz::R
end

get_order(ritz::R, which::T) where {T<:Target,R<:RitzValues} = EigenvalueOrder{T,R}(ritz)

lt(o::EigenvalueOrder{LM}, i::Integer, j::Integer) = 
    @inbounds return abs(o.ritz.λs[j]) < abs(o.ritz.λs[i])
lt(o::EigenvalueOrder{LR}, i::Integer, j::Integer) = 
    @inbounds return real(o.ritz.λs[j]) < real(o.ritz.λs[i])
lt(o::EigenvalueOrder{LI}, i::Integer, j::Integer) =
    @inbounds return imag(o.ritz.λs[j]) < imag(o.ritz.λs[i])
lt(o::EigenvalueOrder{SR}, i::Integer, j::Integer) = 
    @inbounds return real(o.ritz.λs[i]) < real(o.ritz.λs[j])
lt(o::EigenvalueOrder{SI}, i::Integer, j::Integer) =
    @inbounds return imag(o.ritz.λs[i]) < imag(o.ritz.λs[j])