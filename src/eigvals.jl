"""
    eigvalues(A::AbstractMatrix{T}, tol = eps()) -> Vector{complex(T)}

Computes the eigenvalues of the matrix A. Assumes that A is quasi-upper triangular.
The eigenvalues are returned in complex arithmetic, even if their imaginary
part is 0.
"""
function eigvalues(A::AbstractMatrix{T}; tol = eps(real(T))) where {T}
    n = size(A, 1)
    λs = Vector{complex(T)}(undef, n)
    i = 1

    @inbounds while i < n
        if is_offdiagonal_small(A, i, tol)
            λs[i] = A[i, i]
            i += 1
        else
            # Conjugate pair
            d = A[i,i] * A[i+1,i+1] - A[i,i+1] * A[i+1,i]
            x = (A[i,i] + A[i+1,i+1]) / real(T)(2)
            y = sqrt(complex(x*x - d))
            λs[i] = x + y
            λs[i + 1] = x - y
            i += 2
        end
    end

    @inbounds if i == n 
        λs[i] = A[n, n] 
    end

    return λs
end

