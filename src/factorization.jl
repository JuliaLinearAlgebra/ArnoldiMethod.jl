"""
Find the largest index `i` such that `H[i, i-1] ≈ 0`. This means that `i:max` constitutes the
active part in the Arnoldi decomp, while `V[:, 1:i-1]` forms a basis for an invariant 
subspace of A. Sets H[i, i-1] := 0.
"""
function detect_convergence!(H::AbstractMatrix{T}, tolerance) where {T}
    n = size(H, 2)
    #λs, xs = eig(view(H, 1 : n, 1 : n))
    #perm = sortperm(λs, by = abs, rev = true)
    #λs = λs[perm]
    #xs = view(xs, :, perm)

    # @inbounds for i = n-1 : -1 : 1
    #     if abs(H[i + 1, i]) * abs(xs[i,i]) < max(eps(Float64) * norm(view(H, 1:i, 1:i)), tolerance * abs(λs[i]))
    #         H[i + 1, i] = zero(T)
    #         return i + 1
    #     end
    # end

    @inbounds for i = n : -1 : 2
        if abs(H[i, i-1]) ≤ 1e-6 # tolerance
            H[i, i-1] = zero(T)
            return i
        end
    end

    # No convergence :(
    return 1
end

