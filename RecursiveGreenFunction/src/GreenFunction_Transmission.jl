"""
    green_function_transmission(Gamma_Left, Gamma_Right, G_Centre_Ret)

Compute electron transmission via the Fisher-Lee formula:
  T = Tr[Γ_L G^ret Γ_R G^adv]

# Arguments
- `Gamma_Left::AbstractMatrix`: Level-width matrix of the left lead.
- `Gamma_Right::AbstractMatrix`: Level-width matrix of the right lead.
- `G_Centre_Ret::AbstractMatrix`: Retarded Green function G_C^ret(ω).

# Returns
- `T::ComplexF64`: Raw trace value. Use `real(T)` for the physical transmission.

# References
Fisher & Lee, Phys. Rev. B 23, 6851 (1981).
"""
function green_function_transmission(Gamma_Left::AbstractMatrix,
                                     Gamma_Right::AbstractMatrix,
                                     G_Centre_Ret::AbstractMatrix)
    return tr(Gamma_Left * G_Centre_Ret * Gamma_Right * G_Centre_Ret')
end
