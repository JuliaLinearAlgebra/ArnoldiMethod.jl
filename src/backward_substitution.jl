using IRAM: is_offdiagonal_small

function backward_subst!(R::AbstractMatrix{T}, λ::Number, y::AbstractVector, tol=100eps(real(T))) where{T}

    n = size(R,1)

    @inbounds for i = n : -1 : 1
        if abs(R[i,i] - λ) < tol
            if abs(y[i]) < tol
                y[i] = zero(T)
            else
                y[i] /= tol
                y .*= tol
            end
        else
            y[i] /= R[i,i] - λ
        end
        # R[i,i] = one(T)
        @inbounds for k = i-1 : -1 : 1
            y[k] -= y[i]*R[k,i]
            # R[k,i] = zero(T)
        end
    end
end

function backward_subst!(R::AbstractMatrix{T}, λ::Number, y::AbstractVector, tol=100eps(T)) where{T<:Real}

    n = size(R,1)
    i = n

    while i > 1
        if !is_offdiagonal_small(R, i-1, tol)
            det = (R[i-1,i-1]-λ)*(R[i,i]-λ) - R[i,i-1]*R[i-1,i]
            a1 = ((R[i,i]-λ) * y[i-1] - R[i-1,i] * y[i]) / det
            a2 = (-R[i,i-1] * y[i-1] + (R[i-1,i-1]-λ) * y[i]) / det
            y[i-1] = a1
            y[i] = a2

            @inbounds for k = i-2 : -1 : 1
                y[k] -= y[i-1]*R[k,i-1]
                y[k] -= y[i]*R[k,i]
            end
            i-=2
        else
            # @show abs(R[i,i] - λ)
            if abs(R[i,i] - λ) < tol
                if abs(y[i]) < tol
                    y[i] = zero(T)
                else
                    y[i] /= tol
                    y .*= tol
                end
                # temp = y[i]
                # y .*= tol
                # y .*= 0
                # y[i] = temp
            else
                y[i] /= R[i,i] - λ
            end
            @inbounds for k = i-1 : -1 : 1
                y[k] -= y[i]*R[k,i]
            end
            i-=1
        end
    end
    if i==1
        if abs(R[i,i] - λ) < tol
            if abs(y[i]) < tol
                y[i] = zero(T)
            else
                y[i] /= tol
                y .*= tol
            end
        else
            y[i] /= R[i,i] - λ
        end
        @inbounds for k = i-1 : -1 : 1
            y[k] -= y[i]*R[k,i]
        end
        i-=1
    end
end


function backward_subst!(R::AbstractMatrix{TR}, y::AbstractVector{Ty}, tol::Number=100eps(TR)) where{TR<:Real, Ty}

    n = size(R,1)
    y .= zero(Ty)

    A = view(R, n-1:n, n-1:n)

    det = A[1,1]*A[2,2] - A[2,1]*A[1,2]
    tr = A[1,1] + A[2,2]
    λ = 0.5*(tr + sqrt(Complex(tr*tr - 4*det)))

    x1 = -A[1,2] / (A[1,1] - λ)
    @assert isapprox(x1, -(A[2,2] - λ) / A[2,1])

    y[n-1] = x1
    y[n] = one(Ty)

    @inbounds for k = n-2 : -1 : 1
        y[k] -= y[n-1]*R[k,n-1]
        y[k] -= y[n]*R[k,n]
    end

    i = n - 2
    while i > 1
        if !is_offdiagonal_small(R, i-1, tol)
            det = (R[i-1,i-1]-λ)*(R[i,i]-λ) - R[i,i-1]*R[i-1,i]
            a1 = ((R[i,i]-λ) * y[i-1] - R[i-1,i] * y[i]) / det
            a2 = (-R[i,i-1] * y[i-1] + (R[i-1,i-1]-λ) * y[i]) / det
            y[i-1] = a1
            y[i] = a2

            @inbounds for k = i-2 : -1 : 1
                y[k] -= y[i-1]*R[k,i-1]
                y[k] -= y[i]*R[k,i]
            end
            i-=2
        else
            # @show abs(R[i,i] - λ)
            if abs(R[i,i] - λ) < tol
                if abs(y[i]) < tol
                    y[i] = zero(Ty)
                else
                    y[i] /= tol
                    y[:]*= tol
                end
                # temp = y[i]
                # y[:]*= tol
                # y[:]*= 0
                # y[i] = temp
            else
                y[i] /= R[i,i] - λ
            end
            @inbounds for k = i-1 : -1 : 1
                y[k] -= y[i]*R[k,i]
            end
            i-=1
        end
    end
    if i==1
        if abs(R[i,i] - λ) < tol
            if abs(y[i]) < tol
                y[i] = zero(Ty)
            else
                y[i] /= tol
                y[:].*= tol
            end
        else
            y[i] /= R[i,i] - λ
        end
        @inbounds for k = i-1 : -1 : 1
            y[k] -= y[i]*R[k,i]
        end
        i-=1
    end
end