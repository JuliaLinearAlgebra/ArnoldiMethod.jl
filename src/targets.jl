using Base.Order: Ordering
import Base.Order: lt

# Here is just some helper functions to get type stable order functors, cause I don't like
# what Julia base has to offer at the moment.

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

struct OrderPerm{TV<:AbstractVector,O<:Ordering} <: Ordering
    xs::TV
    ord::O
end

struct OrderLt{Tf} <: Ordering
    less::Tf
end

struct OrderBy{Tf,O<:Ordering} <: Ordering
    f::Tf
    ord::O
end

struct OrderReverse{O<:Ordering} <: Ordering
    ord::O
end

const Forward = OrderLt(isless)
const Backward = OrderReverse(Forward)

OrderBy(f) = OrderBy(f, Forward)

@inline lt(o::OrderLt, a, b) = o.less(a, b)
@inline lt(o::OrderBy, a, b) = lt(o.ord, o.f(a), o.f(b))
@inline lt(o::OrderReverse, a, b) = lt(o.ord, b, a)

@inline function lt(o::OrderPerm, i::Integer, j::Integer)
    @inbounds fst = o.xs[i]
    @inbounds snd = o.xs[j]

    # First compare by value, then by key to ensure stable order.
    return lt(o.ord, fst, snd) ? true : (lt(o.ord, snd, fst) ? false : i < j)
end


# Helper functions
get_order(ritz::RitzValues, which::LM) = OrderPerm(ritz.λs, OrderReverse(OrderBy(abs)))
get_order(ritz::RitzValues, which::LR) = OrderPerm(ritz.λs, OrderReverse(OrderBy(real)))
get_order(ritz::RitzValues, which::SR) = OrderPerm(ritz.λs, OrderBy(real))
get_order(ritz::RitzValues, which::LI) = OrderPerm(ritz.λs, OrderReverse(OrderBy(imag)))
get_order(ritz::RitzValues, which::SI) = OrderPerm(ritz.λs, OrderBy(imag))
