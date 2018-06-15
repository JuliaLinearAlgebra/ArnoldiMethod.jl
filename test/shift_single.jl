using Base.Test

using IRAM: single_shift!

@testset "Single Shifted QR" begin
    max = 20
    min = 10

    # Test whether the eigenvalues of the H[1:m,1:m] block are equal to the original
    # eigenvalues of H with some selected eignvalues removed.
    for T in (Complex128,)
        # Generate a Hessenberg matrix
        H = triu(rand(T, max + 1, max), -1)
        H_square = view(H, 1 : max, 1 : max)
        λs = sort!(eigvals(H_square), by = abs, rev = true)
        
        for m = max : -1 : min + 1
            μ = λs[m]
            H_small = view(H, 1:m+1, 1:m)
            single_shift!(H_small, μ, debug = false)
            H_square_small = view(H, 1:m-1, 1:m-1)
            θs = eigvals(H_square_small)
            @test λs[1:m-1] ≈ sort!(θs, by = abs, rev = true)
        end
    end
end
