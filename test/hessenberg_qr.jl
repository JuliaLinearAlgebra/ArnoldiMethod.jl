using Base.Test
using IRAM: qr!, Hessenberg, mul!, ListOfRotations

@testset "QR on Hessenberg matrix" begin
    for T in (Float64, Complex128)
        # Random Hessenberg matrix of size 11 x 10.
        H = triu(rand(T, 10, 10), -1)
        R = copy(H)

        list = ListOfRotations(T, 9)
        qr!(Hessenberg(R), list)
        Q = mul!(eye(T, 10), list)

        @test H ≈ Q * R
        # @test istriu(H)
    end
end