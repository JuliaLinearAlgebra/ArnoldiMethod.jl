using Base.Test
using IRAM: qr!, Hessenberg, mul!

@testset "QR on Hessenberg matrix" begin
    # Random Hessenberg matrix of size 11 x 10.
    H = triu(rand(11, 10), -1)
    R = copy(H)

    rotations = qr!(Hessenberg(R))
    Q = mul!(eye(11), rotations)

    @test H ≈ Q * R
    # @test istriu(H)
end