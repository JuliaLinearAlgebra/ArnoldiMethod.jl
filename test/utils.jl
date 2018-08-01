using LinearAlgebra

"""
    normal_hessenberg_matrix(Float64, vals::AbstractVector)

Generate a normal hessenberg matrix with eigenvalues `vals`
"""
function normal_hessenberg_matrix(T::Type{<:Number}, vals::AbstractVector)
    n = length(vals)
    Q, R = qr(randn(T, n, n))
    A = Q * Diagonal(vals) * Q'
    return triu(hessenberg!(A).factors, -1)
end

function normal_hessenberg_matrix_conjugate_pair(T::Type{<:Real}, vals::AbstractVector{<:Complex})
    n = length(vals)
    Q, R = qr(randn(T, n, n))
    D = zeros(T, n, n)
    i = 1
    while i ≤ n
        if imag(vals[i]) != 0
            //
        else

        end
    end
    A = Q * Diagonal(vals) * Q'
    return triu(hessenberg!(A).factors, -1)
end

"""
    realimag(1 + 3im) -> (1, 3)

Split imaginary number into a tuple of real and imaginary part
"""
realimag(x) = (real(x), imag(x))

"""
    is_hessenberg(H) -> bool

Test whether the sub-subdiagonals of H are zero.
"""
is_hessenberg(H) = norm(tril(H, -2)) == 0