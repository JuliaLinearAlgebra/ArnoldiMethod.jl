"""
Run IRAM until the eigenvectors are approximated to the prescribed tolerance or until 
`max_restarts` has been reached.
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, nev = min, ε = eps(T), max_restarts = 10) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : min)

    # min′ is the effective starting point -- may be min - 1 when the last two removed guys
    # are a complex conjugate
    min′ = min

    active = 1
    V_prealloc = Matrix{T}(undef, n, min)
    for restarts = 1 : max_restarts
        n = max - active + 1

        iterate_arnoldi!(A, arnoldi, min′ + 1 : max)
        
        # Compute shifts
        λs = compute_shifts(arnoldi.H, active, max)

        min′ = implicit_restart!(arnoldi, λs, min, max, active, V_prealloc)
        new_active = detect_convergence!(view(arnoldi.H, active:min′+1, active:min′), ε)
        new_active += active - 1 
        if new_active > active + 1
            # Bring the new locked part oF H into upper triangular form
            transform_converged(arnoldi, active, new_active, min′, V_prealloc)
        end

        active = new_active

        active > nev && break
    end

    return PartialSchur(arnoldi.V, arnoldi.H, active - 1)
end

"""
Transfrom the converged block into an upper triangular form.
"""
function transform_converged(arnoldi, active, new_active, min′, V_prealloc)
    
    # H = Q R Q'

    # A V = V H
    # A V = V Q R Q'
    # A (V Q) = (V Q) R
    
    # V <- V Q
    # H_right <- Q' H_right
    # H_lock <- Q' H_lock Q
    # H_above <- H_above Q

    Q_large = Matrix{eltype(arnoldi.H)}(I, new_active - 1, new_active - 1)

    H_locked = view(arnoldi.H, active : new_active - 1, active : new_active - 1)
    H_copy = copy(H_locked)

    H_copy_full = copy(arnoldi.H)

    local_schurfact!(arnoldi.H, active, new_active - 1, Q_large)
    Q_small = view(Q_large, active : new_active - 1, active : new_active - 1)

    V_locked = view(arnoldi.V, :, active : new_active - 1)
    mul!(view(V_prealloc, :, active : new_active - 1), V_locked, Q_small)
    V_locked .= view(V_prealloc, :, active : new_active - 1)

end

function compute_shifts(H::AbstractMatrix{T}, active, max) where {T}
    n = max - active + 1
    # Compute the eigenvalues of the active part
    Q = Matrix{T}(I, n, n)
    # vals,vecs = eigen(H[active:max,active:max])

    R = copy(view(H, active:max, active:max))
    local_schurfact!(R, Q)
    # @assert isapprox(H[active:max,active:max]*Q, Q*R)

    # rval, rvec = eigen(R)
    λs = eigvalues(R)

    y = Vector{T}(undef,n)
    res = Vector{Float64}(undef,n)
    # ys = Matrix{T}(undef, n,n)
    @inbounds for i = n : -1 : 1
        y[i] = one(T)
        y[1:i-1] .= - view(R, 1:i-1, i)
        y[i+1:n] .= zero(T)
        backward_subst!(view(R,1:i-1,1:i-1), y, R[i,i])
        y ./= norm(y)
        # ys[:,i] .= y
        res[i] = abs(transpose(view(Q, n, 1:n))*y * H[max + 1, max])
    end
    # @show res
    perm = sortperm(res, by=abs)

    # perm_r = sortperm(rval, by=abs)
    # perm_y = sortperm(diag(R), by=abs)
    # perm_vals = sortperm(vals, by=abs)
    # @assert isapprox(sort(diag(R), by=abs), sort(vals, by=abs))
    # ys_perm = ys[:, perm_y]
    # rvec_perm = rvec[:, perm_r]
    # vecs_perm = vecs[:,perm_vals]
    # @assert isapprox(abs.(ys_perm), abs.(rvec_perm))

    # @show ys_perm[:, 4]
    # @show rvec_perm[:, 4]
    # a = ys_perm - rvec_perm
    # @show norm(a[:,1:end])
    # display(a)
    # @assert isapprox(H[active:max,active:max]*Q, Q*R)
    # @assert isapprox(ys_perm,rvec_perm)
    # @show norm(H[active:max,active:max]*Q - Q*R)
    # @show norm(H[active:max,active:max] * vecs - vecs * Diagonal(vals))
    # @show norm(H[active:max,active:max] * vecs_perm - vecs_perm * Diagonal(vals[perm_vals]))
    # @show norm(Q*R*Q' * vecs_perm - vecs_perm * Diagonal(vals[perm_vals]))
    # @show norm(R* Q'*vecs_perm - Q'*vecs_perm * Diagonal(vals[perm_vals]))
    # @show norm(R * ys_perm - ys_perm * Diagonal(vals[perm_vals]))
    # for i = active:max
    #     # @show norm(Q'*vecs_perm[:,i] - ys_perm[:,i])
    #     # display(Q'*vecs_perm[:,i])
    #     display(abs.(vecs_perm[:,i]))
    #     @show abs(vecs_perm[n,i])
    #     temp = vals[perm_vals]
    #     # @show norm(R*Q'*vecs_perm[:,i] - temp[i]*Q'*vecs_perm[:,i])
    #     @show norm(H[active:max,active:max]*vecs_perm[:,i] - temp[i]*vecs_perm[:,i])
    #     # display(ys_perm[:,i])
    #     display(abs.(Q*ys_perm[:,i]))
    #     @show abs(dot(conj(Q[n,1:n]),ys_perm[:,i]))
    #     tempor = Q*ys_perm[:,i]
    #     @show abs(tempor[n])
    #     @show norm(H[active:max,active:max]*Q*ys_perm[:,i] - temp[i]*Q*ys_perm[:,i])
    # end
    # @show norm(Q'*vecs_perm - ys_perm)
    # @show norm(Q*ys_perm - vecs_perm)
    # @show norm(ys_perm), norm(vecs)
    # @show norm(Q*ys_perm), norm(vecs)
    # display(Q' * Q)
    # display(Q*ys[:, perm_y])
    # display(vecs_perm)
    # @info "Residuals" Q*ys_perm - vecs_perm
    # display(ys[:, perm_y])
    # display(rvec_perm)
    λs .= λs[perm]
    return λs

end
