"""
Computes the eigenvalues of the matrix A. Assumes that A is in Schur form.
"""
function eigvalues(A::AbstractMatrix{T}; tol = eps(real(T))) where {T}
        n = size(A, 1)
        λs = Vector{complex(T)}(n)
        i = 1

        while i < n
            if abs(A[i + 1, i]) < tol*(abs(A[i + 1, i + 1]) + abs( A[i, i]))
                λs[i] =  A[i, i]
                i += 1
            else
                d =  A[i, i]*A[i + 1, i + 1] - A[i, i + 1]*A[i + 1, i]
                x = 0.5*(A[i, i] + A[i + 1, i + 1])
                y = sqrt(complex(x*x - d))
                λs[i] = x + y
                λs[i + 1] = x - y
                i += 2
            end
        end

        if i == n 
            λs[i] = A[n, n] 
        end

        return λs
    end

    # H = triu(rand(5,5), -1)
    # tau = Rotation(Base.LinAlg.Givens{Float64}[])
    # singleShiftQR!(HH::StridedMatrix, τ::Rotation, shift::Number, istart::Integer, iend::Integer)