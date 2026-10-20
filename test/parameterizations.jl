using Symbolics 
using LinearAlgebra 

# writing Bernstein basis: 
B(d, i, s) = binomial(d, i) * s^i * (1 - s)^(d - i)

@variables s
B(2, 0, s)

# (d+1) × d forward-difference matrix.
D(d) = diagm(d + 1, d, 0 => -ones(d), -1 => ones(d))
D(2)

# D_d^(k) = D_d * D_{d-1} * ... * D_{d-k+1}.
function Dk(d, k)
    0 <= k <= d || throw(ArgumentError("Require 0 ≤ k ≤ d"))

    M = Matrix{Float64}(I, d + 1, d + 1)
    for j in d:-1:(d - k + 1)
        M = M * D(j)
    end
    return M
end
Dk(2,1)

function gamma_derivative(Γ, s, k)
    d = size(Γ, 2) - 1
    M = Dk(d, k)
    factor = prod((d - k + 1):d)  # d! / (d-k)!
    return factor * Γ * M * b(d - k, s)
end

Γ = [-2.0 0.0 2.0;
      0.0 1.0 0.0]

gamma_derivative(Γ, 0.5, 0)  # Position:     [0.0, 0.5]
gamma_derivative(Γ, 0.5, 1)  # 1st derivative: [4.0, 0.0]
gamma_derivative(Γ, 0.5, 2)  # 2nd derivative: [0.0, -4.0]