using Test
using IRAM: qr!, Hessenberg, ListOfRotations

@testset "QR on Hessenberg matrix" begin
    for T in (Float64, ComplexF64)
        # Random Hessenberg matrix of size 11 x 10.
        H = triu(rand(T, 10, 10), -1)
        R = copy(H)

        list = ListOfRotations(T, 9)
        qr!(Hessenberg(R), list)
        Q = rmul!(Matrix{T}(I, 10, 10), list)

        @test H ≈ Q * R
        # @test istriu(H)
    end
end