"""
Run IRAM until the eigenvectors are approximated to the prescribed tolerance or until 
`max_restarts` has been reached.
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, nev = min, ε = eps(T), max_restarts = 10) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    h = Vector{T}(max)
    iterate_arnoldi!(A, arnoldi, 1 : min, h)

    # min′ is the effective starting point -- may be min - 1 when the last two removed guys
    # are a complex conjugate
    min′ = min

    active = 1
    V_prealloc = Matrix{T}(n, min)
    for restarts = 1 : max_restarts

        iterate_arnoldi!(A, arnoldi, min′ + 1 : max, h)

        # λs = sort!(eigvals(view(arnoldi.H, active:max, active:max)), by = abs, rev = true)
        schur = schurfact(view(arnoldi.H, active:max, active:max))
        λs = sort!(eigvalues(schur[:T]), by = abs, rev = true)

        min′ = implicit_restart!(arnoldi, λs, min, max, active, V_prealloc)
        new_active = detect_convergence!(view(arnoldi.H, active:min′+1, active:min′), ε)
        new_active += active - 1 
        # @show vecnorm(A * arnoldi.V[:, 1:min′] - arnoldi.V[:, 1:min′+1] * arnoldi.H[1:min′+1, 1:min′] )
        if new_active > active + 1
            # Bring the new locked part oF H into upper triangular form
            # display(view(arnoldi.H, 1 : new_active - 1, 1 : new_active - 1))
            # @show sort!(eigvalues(view(arnoldi.H, 1 : new_active - 1, 1 : new_active - 1)), by = abs, rev = true)
            # @show sort!(eigvals(view(arnoldi.H, 1 : new_active - 1, 1 : new_active - 1)), by = abs, rev = true)
            transform_converged(arnoldi, active, new_active, min′, V_prealloc)
        end

        active = new_active

        @show active

        if active > nev
            break 
        end
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

    H_locked = view(arnoldi.H, active : new_active - 1, active : new_active - 1)
    H_copy = copy(H_locked)
    # schur_form = schurfact(H_locked)
    # display(schur_form.Z)
    # display(H_locked)
    H_copy_full = copy(arnoldi.H)

    _ , Q_large = schurfact!(arnoldi.H, active, new_active - 1, shiftmethod = :Rayleigh)
    Q_small = view(Q_large, active : new_active - 1, active : new_active - 1)
    display(H_locked)
    @show sort!(eigvalues(H_locked), by = abs, rev = true)
    @show sort!(eigvals(H_locked), by = abs, rev = true)


    V_locked = view(arnoldi.V, :, active : new_active - 1)
    # H_right = view(arnoldi.H, active : new_active - 1, active : min′)

    A_mul_B!(view(V_prealloc, :, active : new_active - 1), V_locked, Q_small)
    V_locked .= view(V_prealloc, :, active : new_active - 1)
    # @show vecnorm(schur_form.Z' * H_copy * schur_form.Z - schur_form[:Schur])
    @show vecnorm(Q_small' * H_copy * Q_small - H_locked)
    # display(Q_small)
    # Ac_mul_B!(H_right, schur_form.Z, copy(H_right))
    
    # H_above = view(arnoldi.H, 1 : new_active - 1, active : new_active - 1)
    # A_mul_B!(H_above, copy(H_above), schur_form.Z)

end
