using Test
using ArnoldiMethod: sylv
using StaticArrays

@testset "Tiny sylvester equation $T, $s" for T in (Float64, ComplexF64), s = ((2,2),(2,1),(1,2))
    p, q = s
    A = @SMatrix rand(T, p, p)
    B = @SMatrix rand(T, q, q)
    C = @SMatrix rand(T, p, q)

    X = sylv(A, B, C)

    @test A * X - X * B ≈ C
end