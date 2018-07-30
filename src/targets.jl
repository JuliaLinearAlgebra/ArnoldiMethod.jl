abstract type Target end

# For finding eigenvalues with the smallest magnitude
struct SM <: Target end

# For finding eigenvalues with the largest magnitude
struct LM <: Target end

# For finding eigenvalues with the largest real part
struct LR <: Target end

# For finding eigenvalues with the smallest real part
struct SR <: Target end

# For finding eigenvalues with the largest imaginary part
struct LI <: Target end

# For finding eigenvalues with the smallest imaginary part
struct SI <: Target end

sort_vals!(λs,::LM) = sort!(λs, by=abs, rev=true)

sort_vals!(λs,::SM) = sort!(λs, by=abs)

sort_vals!(λs,::LR) = sort!(λs, by = real)

sort_vals!(λs,::SR) = sort!(λs, by = f(x) = - real(x))

sort_vals!(λs,::LI) = sort!(λs, by = imag)

sort_vals!(λs,::SI) = sort!(λs, by = f(x) = - imag(x))