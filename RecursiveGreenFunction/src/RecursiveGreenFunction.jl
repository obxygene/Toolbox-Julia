"""
RecursiveGreenFunction

Julia package for Non-Equilibrium Green Function (NEGF) calculations.
Implements the recursive Green function method for quantum transport in
tight-binding models using the Landauer-Büttiker formalism.

# Methods
- Lopez-Sancho iterative surface Green function
- Recursive layer-by-layer Dyson equation
- Fisher-Lee transmission formula T = Tr[ΓL G ΓR G†]
- Six-terminal Hall-bar geometry
- Block-tridiagonal matrix inversion
- Krylov-based matrix exponential (Lanczos)
- Finite-temperature Sommerfeld convolution
"""
module RecursiveGreenFunction

using LinearAlgebra
using SparseArrays

include("SurfaceGreenFunction.jl")
include("SurfaceGreenFunction_SelfEnergy.jl")
include("SurfaceGreenFunction_Broadening.jl")
include("SurfaceGreenFunction_Gcc_1L_AddEndLayer.jl")
include("RecursiveGreenFunction_1L.jl")
include("GreenFunction_Transmission.jl")
include("SixTerminal_RGF.jl")
include("tridiag_block_inv.jl")
include("lanczos_expm_apply.jl")
include("FiniteTemperatureTransmission.jl")

"""
    surface_green_function_gcc_1l(HCC, V, layer, omega, Sigma_L, Sigma_R;
                                  eta=1e-5, disorder_type="Clean", Gamma=0.0, Norb=1)

Recursive retarded Green function G_{1L} (layer 1 to layer L).

Compatibility wrapper around `recursive_green_function_1l`.
Returns `(G_1L, nothing)` as a tuple.

# Arguments
- `HCC::AbstractMatrix`: (W×W) intra-layer Hamiltonian.
- `V::AbstractMatrix`: (W×W) inter-layer hopping.
- `layer::Integer`: Number of layers.
- `omega::Number`: Energy.
- `Sigma_L::AbstractMatrix`: Left lead self-energy.
- `Sigma_R::AbstractMatrix`: Right lead self-energy.
- `eta::Real=1e-5`: Infinitesimal broadening.
- `disorder_type::String="Clean"`: "Clean", "Onsite_disorder", "UnitCell_disorder", or "spin_disorder".
- `Gamma::Real=0.0`: Disorder amplitude.
- `Norb::Integer=1`: Orbitals per unit cell (UnitCell_disorder only).

# Returns
- `(G_1L, nothing)`: Retarded GF and placeholder.
"""
function surface_green_function_gcc_1l(HCC::AbstractMatrix, V::AbstractMatrix,
                                       layer::Integer, omega::Number,
                                       Sigma_L::AbstractMatrix, Sigma_R::AbstractMatrix;
                                       eta::Real=1e-5,
                                       disorder_type::String="Clean",
                                       Gamma::Real=0.0,
                                       Norb::Integer=1)
    G_1L = recursive_green_function_1l(HCC, V, layer, omega, Sigma_L, Sigma_R;
                                       eta=eta, disorder_type=disorder_type,
                                       Gamma=Gamma, Norb=Norb)
    return G_1L, nothing
end

export surface_green_function_v2
export surface_green_function_self_energy
export surface_green_function_broadening
export surface_green_function_gcc_1l
export surface_green_function_gcc_1l_add_end_layer
export recursive_green_function_1l
export green_function_transmission
export SigmaLeads, six_terminal_rgf
export tridiag_block_inv
export lanczos_expm_apply
export finite_temperature_transmission

end # module RecursiveGreenFunction
