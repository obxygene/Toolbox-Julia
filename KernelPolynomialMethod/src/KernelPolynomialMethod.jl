"""
KernelPolynomialMethod.jl

Julia port of the MATLAB KPM Toolbox for condensed-matter spectral and
transport calculations.  Translates the following MATLAB functions verbatim:

  KPM_scaleHamiltonian   -> kpm_scale_hamiltonian
  KPM_Kernel             -> kpm_kernel
  KPM_Kernel_Correction  -> kpm_kernel_correction
  KPM_Chebyshev_abscissas-> kpm_chebyshev_abscissas
  KPM_rescale            -> kpm_rescale
  KPM_fermi_distribution -> kpm_fermi_distribution
  KPM_Moments_Spectrum   -> kpm_moments_spectrum
  KPM_Moments_Operator   -> kpm_moments_operator
  KPM_Moments_Correlator -> kpm_moments_correlator
  KPM_DCT                -> kpm_dct
  KPM_DOS                -> kpm_dos
  KPM_Correlator_basis   -> kpm_correlator_basis
  KPM_Hall_Conductance   -> kpm_hall_conductance

References:
  Weisse et al., Rev. Mod. Phys. 78, 275 (2006)
  Garcia et al., Phys. Rev. B 91, 245140 (2015)
"""
module KernelPolynomialMethod

using LinearAlgebra
using SparseArrays
using FFTW
using Arpack
using Plots

include("kpm_scale.jl")
include("kpm_kernel.jl")
include("kpm_moments_spectrum.jl")
include("kpm_moments_operator.jl")
include("kpm_moments_correlator.jl")
include("kpm_dct.jl")
include("kpm_dos.jl")
include("kpm_conductivity.jl")
include("kpm_plots.jl")

export kpm_scale_hamiltonian
export kpm_kernel, kpm_kernel_correction
export kpm_chebyshev_abscissas, kpm_rescale, kpm_fermi_distribution
export kpm_moments_spectrum, kpm_moments_operator, kpm_moments_correlator
export kpm_dct
export kpm_dos
export kpm_correlator_basis, kpm_hall_conductance
export plot_dos, plot_dos_compare
export plot_conductance, plot_conductance_compare, plot_convergence

end # module
