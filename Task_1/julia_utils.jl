"""
Julia implementation for Task #1 using Yao.jl + Optim.jl.
"""

using LinearAlgebra
using Random
using Yao
using Yao.Blocks
using Optim

const N_QUBITS = 4
const PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

function random_phi(; seed::Int=101, n_qubits::Int=N_QUBITS)
    Random.seed!(seed)
    dim = 2^n_qubits
    φ = 2π .* rand(dim) .+ (2π .* rand(dim)) .* im
    φ ./ norm(φ)
end

function split_angle_vector(angles::AbstractVector{<:Real})
    if isempty(angles) || length(angles) % 8 != 0
        throw(ArgumentError("angles length must be a non-zero multiple of 8"))
    end
    odd = [angles[i:i+3] for i in 1:8:length(angles)]
    even = [angles[i:i+3] for i in 5:8:length(angles)]
    layers = length(angles) ÷ 8
    return odd, even, layers
end

rotation(axis::Symbol, θ::Real, q::Int) =
    axis === :x ? put(N_QUBITS, q => Rx(θ)) :
    axis === :y ? put(N_QUBITS, q => Ry(θ)) :
    axis === :z ? put(N_QUBITS, q => Rz(θ)) :
    throw(ArgumentError("unsupported axis: $axis"))

function controlled_pair(axis::Symbol, control::Int, target::Int)
    g = axis === :x ? X : axis === :y ? Y : axis === :z ? Z : nothing
    g === nothing && throw(ArgumentError("unsupported axis: $axis"))
    return control(control, target => g)
end

function build_odd_block(axis::Symbol, layer_angles)
    length(layer_angles) == 4 || throw(ArgumentError("odd block requires 4 angles"))
    chain(N_QUBITS, [rotation(axis, layer_angles[q], q) for q in 1:4])
end

function build_even_block(axis::Symbol, layer_angles)
    length(layer_angles) == 4 || throw(ArgumentError("even block requires 4 angles"))
    rotations = [rotation(axis, layer_angles[q], q) for q in 1:4]
    entanglers = [controlled_pair(axis, c, t) for (c, t) in PAIRS]
    chain(N_QUBITS, vcat(rotations, entanglers))
end

const CASE_AXES = Dict(
    1 => (:x, :x), 2 => (:x, :y), 3 => (:x, :z),
    4 => (:y, :x), 5 => (:y, :y), 6 => (:y, :z),
    7 => (:z, :x), 8 => (:z, :y), 9 => (:z, :z),
)

function build_case(case_num::Int, odd_angles, even_angles)
    haskey(CASE_AXES, case_num) || throw(ArgumentError("case_num must be 1..9"))
    length(odd_angles) == length(even_angles) || throw(ArgumentError("odd/even layers mismatch"))
    odd_axis, even_axis = CASE_AXES[case_num]
    blocks = Any[]
    for layer in eachindex(odd_angles)
        push!(blocks, build_odd_block(odd_axis, odd_angles[layer]))
        push!(blocks, build_even_block(even_axis, even_angles[layer]))
    end
    return chain(N_QUBITS, blocks)
end

function state_from_circuit(circuit)
    reg = zero_state(N_QUBITS)
    out = circuit * reg
    return statevec(out)
end

function objective_case(angles::AbstractVector{<:Real}, case_num::Int, φ)
    odd, even, _ = split_angle_vector(angles)
    ψ = state_from_circuit(build_case(case_num, odd, even))
    return norm(ψ - φ)
end

function flatten_angles(layer::Int, odd_block_angles, even_block_angles)
    layer >= 1 || throw(ArgumentError("layer must be >= 1"))
    req = layer
    length(odd_block_angles) >= req || throw(ArgumentError("insufficient odd layers"))
    length(even_block_angles) >= req || throw(ArgumentError("insufficient even layers"))

    out = Float64[]
    for k in 1:layer
        append!(out, odd_block_angles[k])
        append!(out, even_block_angles[k])
    end
    return out
end

function reshape_angles(opt_vector::AbstractVector{<:Real})
    odd, even, _ = split_angle_vector(opt_vector)
    return odd, even
end

function optimize_case(layer::Int, odd_block_angles, even_block_angles, case_num::Int, φ)
    initial = flatten_angles(layer, odd_block_angles, even_block_angles)
    lower = zeros(length(initial))
    upper = fill(2π, length(initial))

    f = θ -> objective_case(θ, case_num, φ)
    result = optimize(f, lower, upper, initial, Fminbox(LBFGS()))
    odd, even = reshape_angles(Optim.minimizer(result))
    return odd, even, result
end
