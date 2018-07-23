using Test

using IRAM: eigvalues, local_schurfact!, backward_subst!, compute_shifts, initialize, iterate_arnoldi!
using LinearAlgebra

@testset "Shift computation" begin
    n = 20
    # A = rand(ComplexF64, n,n)
    A = triu(rand(ComplexF64, n,n))
    for i = 1:n
        A[i,i] = i
    end
    max = 10
    h = Vector{ComplexF64}(undef, max)
    arnoldi = initialize(ComplexF64, n, max)
    iterate_arnoldi!(A, arnoldi, 1:max, h)

    shifts = compute_shifts(arnoldi.H, 1, max)
    vals, vecs = eigen(arnoldi.H[1:max, 1:max])

    # Test whether the shifts are the eigenvalues of H
    @test sort(vals, by=abs) ≈ sort(shifts, by=abs)

    res = Vector{Float64}(undef,max)
    for i = 1:max
        res[i] = norm(A * arnoldi.V[:,1:max] * vecs[:,i] - vals[i] * arnoldi.V[:,1:max] * vecs[:,i])
        # @show vecs[:,i]
    end
    @show res
    perm = sortperm(res, by=abs)
    vals = vals[perm]

    # Test the order of the shifts
    @test abs.(vals) ≈ abs.(shifts)
    @test vals ≈ shifts
end

# AV = VH
# A x = lambda x
# VHV' x = lambda x + res
# HV' x = lambda V' x + res

# HQ = QR
# H x = lambda x
# QRQ' x = lambda x
# RQ' x = lambda Q' x