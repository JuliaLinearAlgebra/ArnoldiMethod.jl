# Tests in-place products with the implicit Given's rotations we have by forming
# the Given's rotation as a matrix explicitly.

using Test, LinearAlgebra
using ArnoldiMethod: Hessenberg, Rotation2, Rotation3

@testset "Givens rotation" begin
    @testset "Single rotation" begin
        @testset "lmul!" for T in (Float64, ComplexF64)
            A = rand(T, 6, 5)

            G = Rotation2(rand(real(T)), rand(T), 2)
            G_mat = Matrix(G, 6)

            @test [A[:,1:1] G_mat * A[:,2:4] A[:,5:5]] ≈ lmul!(G, copy(A), 2, 4)
            @test G_mat * A ≈ lmul!(G, copy(A))
        end

        @testset "rmul!" for T in (Float64, ComplexF64)
            A = rand(T, 10, 5)

            G = Rotation2(rand(real(T)), rand(T), 2)
            G_mat = Matrix(G, 5)

            @test [A[1:1,:]; A[2:4,:]*G_mat'; A[5:10,:]] ≈ rmul!(copy(A), G, 2, 4)
            @test A * G_mat' ≈ rmul!(copy(A), G)
        end
    end

    @testset "Double rotation" begin
        @testset "lmul!" for T in (Float64, ComplexF64)
            A = rand(T, 6, 5)

            G = Rotation3(rand(real(T)), rand(T), rand(real(T)), rand(T), 2)
            G_mat = Matrix(G, 6)

            @test [A[:,1:1] G_mat * A[:,2:4] A[:,5:5]] ≈ lmul!(G, copy(A), 2, 4)
            @test G_mat * A ≈ lmul!(G, copy(A))
        end

        @testset "rmul!" for T in (Float64, ComplexF64)
            A = rand(T, 10, 5)

            G = Rotation3(rand(real(T)), rand(T), rand(real(T)), rand(T), 2)
            G_mat = Matrix(G, 5)

            @test [A[1:1,:]; A[2:4,:]*G_mat'; A[5:10,:]] ≈ rmul!(copy(A), G, 2, 4)
            @test A * G_mat' ≈ rmul!(copy(A), G)
        end
    end
end