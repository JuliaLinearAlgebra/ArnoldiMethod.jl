using Base.Test

using IRAM: double_shift!

function generate_real_H_with_imaginary_eigs(n, T::Type = Float64)
    while true
        H = triu(rand(T, n + 1, n), -1)
        λs = sort!(eigvals(view(H, 1 : n, 1 : n)), by = abs)

        for i = 1 : n
            μ = λs[i]
            if imag(μ) != 0
                # Remove conjugate pair
                deleteat!(λs, (i, i + 1))
                return H, λs, μ
            end
        end
    end
end

@testset "Double Shifted QR" begin
    n = 20

    # Test on a couple random matrices
    for i = 1 : 50
        H, λs, μ = generate_real_H_with_imaginary_eigs(n, Float64)
        H′ = double_shift!(H, μ)
        @test λs ≈ sort!(eigvals(view(H′, 1:n-2, 1:n-2)), by = abs)
    end
end
