using BenchmarkTools

"""
Orthogonalize V[:, k + 1] against V[:, 1 : k] in-place with MGS
"""
function modified_gram_schmidt(V, k)
  v_next = view(V, :, k + 1)
  for i = 1 : k
    v_prev = view(V, :, i)
    h = dot(v_prev, v_next)
    LinAlg.BLAS.axpy!(-h, v_prev, v_next)
  end
  V
end

"""
Orthogonalize V[:, k + 1] against V[:, 1 : k] in-place repeated CGS
"""
function repeated_classical_gram_schmidt(V, k, tmp)
  v_next = view(V, :, k + 1)
  Vk = view(V, :, 1 : k)
  #  @show size(block), size(v_next)

  # tmp = Vk' * v
  # v -= Vk * tmp
  At_mul_B!(tmp, Vk, v_next)
  LinAlg.BLAS.gemv!('N', 1.0, Vk, tmp, -1.0, v_next)
  At_mul_B!(tmp, Vk, v_next)
  LinAlg.BLAS.gemv!('N', 1.0, Vk, tmp, -1.0, v_next)
  V
end

function bench(m = 30)
  tmp = Vector{Float64}(m)

  LinAlg.BLAS.set_num_threads(1)

  V, = qr(rand(100_000, m + 1))
  rand!(view(V, :, m + 1))

  #@benchmark modified_gram_schmidt(A, $m) setup = (A = copy($V)) 
  @benchmark repeated_classical_gram_schmidt(A, $m, B) setup = (A = copy($V); B = copy($tmp))
end