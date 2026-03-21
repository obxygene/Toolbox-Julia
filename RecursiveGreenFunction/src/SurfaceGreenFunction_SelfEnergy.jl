"""
    surface_green_function_self_energy(G_00, H01)

Compute the lead self-energy matrix: Σ = H01 * G₀₀ * H01†.

# Physics Background
In NEGF / Landauer-Büttiker formalism, the retarded self-energy
of a semi-infinite lead encodes the renormalization of the central
Hamiltonian by the leads.

# Arguments
- `G_00::AbstractMatrix`: Surface Green function of the lead.
- `H01::AbstractMatrix`: Coupling Hamiltonian from lead to central region.

# Returns
- `Sigma::Matrix{ComplexF64}`: Retarded self-energy Σ = H01 * G₀₀ * H01†.
"""
function surface_green_function_self_energy(G_00::AbstractMatrix, H01::AbstractMatrix)
    return H01 * G_00 * H01'
end
