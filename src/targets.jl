abstract type Target end

# For finding eigenvalues with the smallest magnitude
struct SM{T} <: Target end

# For finding eigenvalues with the largest magnitude
struct LM{T} <: Target end

# For finding eigenvalues with the largest real part
struct LR{T} <: Target end

# For finding eigenvalues with the smallest real part
struct SR{T} <: Target end

# For finding eigenvalues with the largest imaginary part
struct LI{T} <: Target end

# For finding eigenvalues with the smallest imaginary part
struct SI{T} <: Target end

sort_vals!(λs,::Type{LM}) = sort!(λs, by=abs, rev=true)

sort_vals!(λs,::Type{SM}) = sort!(λs, by=abs)

sort_vals!(λs,::Type{LR}) = sort!(λs, lt = f(a,b) = abs(real(a)) > abs(real(b)))

sort_vals!(λs,::Type{SR}) = sort!(λs, lt = f(a,b) = abs(real(a)) < abs(real(b)))

sort_vals!(λs,::Type{LI}) = sort!(λs, lt = f(a,b) = abs(imag(a)) > abs(imag(b)))

sort_vals!(λs,::Type{SI}) = sort!(λs, lt = f(a,b) = abs(imag(a)) < abs(imag(b)))