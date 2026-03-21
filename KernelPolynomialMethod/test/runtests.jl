using Test
using KernelPolynomialMethod
using LinearAlgebra
using SparseArrays

@testset "KernelPolynomialMethod" begin

    # -------------------------------------------------------
    @testset "kpm_chebyshev_abscissas" begin
        ω = kpm_chebyshev_abscissas(4)
        @test length(ω) == 4
        @test issorted(ω)
        @test all(-1 .< ω .< 1)
        # Standard 4-point nodes
        @test ω ≈ cos.(π .* [3.5, 2.5, 1.5, 0.5] ./ 4) atol=1e-14
    end

    # -------------------------------------------------------
    @testset "kpm_rescale" begin
        ω = kpm_rescale([0.0, 1.0, -1.0], 2.0, 1.0)
        @test ω ≈ [1.0, 3.0, -1.0]
    end

    # -------------------------------------------------------
    @testset "kpm_fermi_distribution" begin
        # T=0: step function
        f = kpm_fermi_distribution([-1.0, 0.0, 1.0], 0.0, 0.0)
        @test f ≈ [1.0, 0.0, 0.0]
        # T>0: correct limit at E=mu
        f2 = kpm_fermi_distribution([0.0], 0.0, 1.0)
        @test f2[1] ≈ 0.5
    end

    # -------------------------------------------------------
    @testset "kpm_kernel" begin
        g = kpm_kernel(4; kernel="Jackson")
        @test length(g) == 4
        @test g[1] ≈ 1.0 atol=1e-12        # g₀ = 1 always
        @test all(0 .< g .<= 1)
        @test issorted(-g)                  # g is decreasing

        g_L = kpm_kernel(10; kernel="Lorentz", lambda=3)
        @test g_L[1] ≈ 1.0 atol=1e-10
        @test all(0 .< g_L .<= 1)
    end

    # -------------------------------------------------------
    @testset "kpm_scale_hamiltonian (1-D chain)" begin
        N = 50
        diag_vals = fill(0.0, N)
        off_diag  = fill(-1.0, N-1)
        H = spdiagm(0 => diag_vals, 1 => off_diag, -1 => off_diag)
        H_t, a, b = kpm_scale_hamiltonian(H; epsilon=0.05)
        @test abs(b) < 0.1      # chain is particle-hole symmetric
        @test a > 0
        # All eigenvalues of H_t should lie in ~(-1, 1)
        λ = eigvals(Matrix(H_t))
        @test all(abs.(λ) .< 1.0 + 1e-6)
    end

    # -------------------------------------------------------
    @testset "kpm_dos (1-D tight-binding chain)" begin
        # 1-D NN chain: DOS should peak near band edges (van Hove)
        N = 200
        off = fill(-1.0, N-1)
        H = spdiagm(1 => off, -1 => off)
        ω, dos = kpm_dos(H; N_randvec=20, N_moments=200, N_points=400, epsilon=0.05)
        @test length(ω) == 400
        @test all(dos .>= 0)
        # Rough integral ≈ 1 (normalised per site)
        dω = ω[2] - ω[1]
        integral = sum(dos) * dω
        @test abs(integral - 1.0) < 0.1
    end

    # -------------------------------------------------------
    @testset "kpm_correlator_basis" begin
        Γ = kpm_correlator_basis(0.3, 8)
        @test size(Γ) == (8, 8)
        # Γ should be Hermitian
        @test Γ ≈ Γ' atol=1e-12
    end

    # -------------------------------------------------------
    @testset "kpm_moments_correlator serial vs parallel" begin
        N = 100
        off = fill(-1.0 + 0im, N-1)
        H = spdiagm(1 => off, -1 => conj.(off))
        H_t, _, _ = kpm_scale_hamiltonian(H; epsilon=0.05)

        # Identity current operators (trivial but checks dimensions)
        J = sparse(I(N) * (1.0 + 0im))

        M_serial   = kpm_moments_correlator(H_t, 5, 16, J, J; parallel=false)
        M_parallel = kpm_moments_correlator(H_t, 5, 16, J, J; parallel=true)

        @test size(M_serial) == (16, 16)
        @test size(M_parallel) == (16, 16)
        # Both should be real-valued for J=I (imaginary part is stochastic noise)
        @test maximum(abs.(imag.(M_serial))) < 1.0
        @test maximum(abs.(imag.(M_parallel))) < 1.0
    end

end # testset
