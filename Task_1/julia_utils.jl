"""
Starter Julia implementation for Task #1.

This file mirrors the Python structure at a high level:
- `build_odd_block!`
- `build_even_block!`
- `build_case!`
- `objective_case`

Dependencies (suggested): Yao.jl, Optim.jl, LinearAlgebra, Random.
"""

using LinearAlgebra
using Random

const N_QUBITS = 4
const PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

"Generate reproducible normalized random target state |ϕ⟩."
function random_phi(; seed::Int=101, n_qubits::Int=N_QUBITS)
    Random.seed!(seed)
    dim = 2^n_qubits
    φ = 2π .* rand(dim) .+ (2π .* rand(dim)) .* im
    φ ./ norm(φ)
end

"Split flat angle vector [odd(4), even(4), ...] into per-layer blocks."
function split_angle_vector(angles::AbstractVector{<:Real})
    odd = [angles[i:i+3] for i in 1:8:length(angles)]
    even = [angles[i:i+3] for i in 5:8:length(angles)]
    layers = length(angles) ÷ 8
    return odd, even, layers
end

"Placeholder for odd block construction (single-qubit rotations)."
function build_odd_block!(circuit, axis::Symbol, layer_angles)
    # TODO: apply Rx/Ry/Rz on all 4 qubits using chosen simulator API.
    return circuit
end

"Placeholder for even block construction (rotations + pairwise controlled gates)."
function build_even_block!(circuit, axis::Symbol, layer_angles)
    # TODO: apply Rx/Ry/Rz on all 4 qubits and pair-wise CX/CY/CZ according to `axis`.
    return circuit
end

const CASE_AXES = Dict(
    1 => (:x, :x), 2 => (:x, :y), 3 => (:x, :z),
    4 => (:y, :x), 5 => (:y, :y), 6 => (:y, :z),
    7 => (:z, :x), 8 => (:z, :y), 9 => (:z, :z),
)

"Build one of the 9 odd/even axis combinations."
function build_case!(circuit, case_num::Int, odd_angles, even_angles)
    odd_axis, even_axis = CASE_AXES[case_num]
    for layer in eachindex(odd_angles)
        build_odd_block!(circuit, odd_axis, odd_angles[layer])
        build_even_block!(circuit, even_axis, even_angles[layer])
    end
    return circuit
end

"Objective: || ψ(θ) - ϕ || (simulator integration TODO)."
function objective_case(angles::AbstractVector{<:Real}, case_num::Int, φ)
    odd, even, _ = split_angle_vector(angles)

    # TODO: initialize circuit using chosen simulator package.
    circuit = nothing
    build_case!(circuit, case_num, odd, even)

    # TODO: run statevector simulation and return norm(state - φ).
    # state = simulate(circuit)
    # return norm(state - φ)

    error("Simulator integration pending: wire this function to Yao.jl (or your preferred backend).")
end
