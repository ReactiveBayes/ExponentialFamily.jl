using ExponentialFamily, Distributions, LinearAlgebra, TinyHugeNumbers
using Test, ForwardDiff, Random, StatsFuns, StableRNGs

import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, getconditioner, compute_logscale, logpartition, basemeasure, insupport,
    sufficientstatistics, fisherinformation, pack_parameters, unpack_parameters, isbasemeasureconstant,
    ConstantBaseMeasure, MeanToNatural, NaturalToMean, NaturalParametersSpace, default_prod_rule, fastcholesky

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type)  = (x,)

function Base.isapprox(a::Tuple, b::Tuple; kwargs...)
    return length(a) === length(b) && all((d) -> isapprox(d[1], d[2]; kwargs...), zip(a, b))
end

function test_exponentialfamily_interface(distribution;
    test_parameters_conversion = true,
    test_similar_creation = true,
    test_distribution_conversion = true,
    test_packing_unpacking = true,
    test_isproper = true,
    test_basic_functions = true,
    test_fisherinformation_properties = true,
    test_fisherinformation_against_hessian = true,
    test_fisherinformation_against_jacobian = true,
    option_assume_no_allocations = false
)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test ef isa ExponentialFamilyDistribution{T}

    test_parameters_conversion && run_test_parameters_conversion(distribution)
    test_similar_creation && run_test_similar_creation(distribution)
    test_distribution_conversion && run_test_distribution_conversion(distribution; assume_no_allocations = option_assume_no_allocations)
    test_packing_unpacking && run_test_packing_unpacking(distribution)
    test_isproper && run_test_isproper(distribution; assume_no_allocations = option_assume_no_allocations)
    test_basic_functions && run_test_basic_functions(distribution; assume_no_allocations = option_assume_no_allocations)
    test_fisherinformation_properties && run_test_fisherinformation_properties(distribution)
    test_fisherinformation_against_hessian && run_test_fisherinformation_against_hessian(distribution; assume_no_allocations = option_assume_no_allocations)
    test_fisherinformation_against_jacobian && run_test_fisherinformation_against_jacobian(distribution; assume_no_allocations = option_assume_no_allocations)

    return ef
end

function run_test_parameters_conversion(distribution)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    tuple_of_θ, conditioner = ExponentialFamily.separate_conditioner(T, params(MeanParametersSpace(), distribution))

    @test all(ExponentialFamily.join_conditioner(T, tuple_of_θ, conditioner) .== params(MeanParametersSpace(), distribution))

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test conditioner === getconditioner(ef)

    # Check the `conditioned` conversions, should work for un-conditioned members as well
    tuple_of_η = MeanToNatural(T)(tuple_of_θ, conditioner)

    @test all(NaturalToMean(T)(tuple_of_η, conditioner) .≈ tuple_of_θ)
    @test all(MeanToNatural(T)(tuple_of_θ, conditioner) .≈ tuple_of_η)
    @test all(NaturalToMean(T)(pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner) .≈ pack_parameters(MeanParametersSpace(), T, tuple_of_θ))
    @test all(MeanToNatural(T)(pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner) .≈ pack_parameters(NaturalParametersSpace(), T, tuple_of_η))

    @test all(map(NaturalParametersSpace() => MeanParametersSpace(), T, tuple_of_η, conditioner) .≈ tuple_of_θ)
    @test all(map(MeanParametersSpace() => NaturalParametersSpace(), T, tuple_of_θ, conditioner) .≈ tuple_of_η)
    @test all(
        map(NaturalParametersSpace() => MeanParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, tuple_of_η), conditioner) .≈
        pack_parameters(MeanParametersSpace(), T, tuple_of_θ)
    )
    @test all(
        map(MeanParametersSpace() => NaturalParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ), conditioner) .≈
        pack_parameters(NaturalParametersSpace(), T, tuple_of_η)
    )

    # Double check the `conditioner` free conversions
    if isnothing(conditioner)
        local _tuple_of_η = MeanToNatural(T)(tuple_of_θ)

        @test all(_tuple_of_η .== tuple_of_η)
        @test all(NaturalToMean(T)(_tuple_of_η) .≈ tuple_of_θ)
        @test all(NaturalToMean(T)(_tuple_of_η) .≈ tuple_of_θ)
        @test all(MeanToNatural(T)(tuple_of_θ) .≈ _tuple_of_η)
        @test all(NaturalToMean(T)(pack_parameters(NaturalParametersSpace(), T, _tuple_of_η)) .≈ pack_parameters(MeanParametersSpace(), T, tuple_of_θ))
        @test all(MeanToNatural(T)(pack_parameters(MeanParametersSpace(), T, tuple_of_θ)) .≈ pack_parameters(NaturalParametersSpace(), T, _tuple_of_η))

        @test all(map(NaturalParametersSpace() => MeanParametersSpace(), T, _tuple_of_η) .≈ tuple_of_θ)
        @test all(map(NaturalParametersSpace() => MeanParametersSpace(), T, _tuple_of_η) .≈ tuple_of_θ)
        @test all(map(MeanParametersSpace() => NaturalParametersSpace(), T, tuple_of_θ) .≈ _tuple_of_η)
        @test all(
            map(NaturalParametersSpace() => MeanParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, _tuple_of_η)) .≈
            pack_parameters(MeanParametersSpace(), T, tuple_of_θ)
        )
        @test all(
            map(MeanParametersSpace() => NaturalParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ)) .≈
            pack_parameters(NaturalParametersSpace(), T, _tuple_of_η)
        )
    end

    @test all(unpack_parameters(NaturalParametersSpace(), T, pack_parameters(NaturalParametersSpace(), T, tuple_of_η)) .== tuple_of_η)
    @test all(unpack_parameters(MeanParametersSpace(), T, pack_parameters(MeanParametersSpace(), T, tuple_of_θ)) .== tuple_of_θ)
end

function run_test_similar_creation(distribution)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test similar(ef) isa ExponentialFamilyDistribution{T}
end

function run_test_distribution_conversion(distribution; assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test @inferred(convert(Distribution, ef)) ≈ distribution

    if assume_no_allocations
        @test @allocated(convert(Distribution, ef)) === 0
    end
end

function run_test_packing_unpacking(distribution)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    tuple_of_θ, conditioner = ExponentialFamily.separate_conditioner(T, params(MeanParametersSpace(), distribution))
    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    tuple_of_η = MeanToNatural(T)(tuple_of_θ, conditioner)

    @test all(unpack_parameters(ef) .≈ tuple_of_η)
    @test @allocated(unpack_parameters(ef)) === 0
end

function run_test_isproper(distribution; assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    exponential_family_form = @inferred(convert(ExponentialFamilyDistribution, distribution))

    @test @inferred(isproper(exponential_family_form))

    if assume_no_allocations
        @test @allocated(isproper(exponential_family_form)) === 0
    end
end

function run_test_basic_functions(distribution; nsamples = 10, assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    # ! do not use `rand(distribution, nsamples)`
    # ! do not use fixed RNG
    samples = [rand(distribution) for _ in 1:nsamples]

    for x in samples
        # We believe in the implementation in the `Distributions.jl`
        @test @inferred(logpdf(ef, x)) ≈ logpdf(distribution, x)
        @test @inferred(pdf(ef, x)) ≈ pdf(distribution, x)
        @test @inferred(mean(ef)) ≈ mean(distribution)
        @test @inferred(var(ef)) ≈ var(distribution)
        @test @inferred(std(ef)) ≈ std(distribution)
        @test rand(StableRNG(42), ef) ≈ rand(StableRNG(42), distribution)
        @test all(rand(StableRNG(42), ef, 10) .≈ rand(StableRNG(42), distribution, 10))
        @test all(rand!(StableRNG(42), ef, [deepcopy(x) for _ in 1:10]) .≈ rand!(StableRNG(42), distribution, [deepcopy(x) for _ in 1:10]))

        @test @inferred(isbasemeasureconstant(ef)) === isbasemeasureconstant(T)
        @test @inferred(basemeasure(ef, x)) == getbasemeasure(T, conditioner)(x)
        @test all(@inferred(sufficientstatistics(ef, x)) .== map(f -> f(x), getsufficientstatistics(T, conditioner)))
        @test @inferred(logpartition(ef)) == getlogpartition(T, conditioner)(η)
        @test @inferred(fisherinformation(ef)) == getfisherinformation(T, conditioner)(η)

        # Double check the `conditioner` free methods
        if isnothing(conditioner)
            @test @inferred(basemeasure(ef, x)) == getbasemeasure(T)(x)
            @test all(@inferred(sufficientstatistics(ef, x)) .== map(f -> f(x), getsufficientstatistics(T)))
            @test @inferred(logpartition(ef)) == getlogpartition(T)(η)
            @test @inferred(fisherinformation(ef)) == getfisherinformation(T)(η)
        end

        # Test that the selected methods do not allocate
        if assume_no_allocations
            @test @allocated(logpdf(ef, x)) === 0
            @test @allocated(pdf(ef, x)) === 0
            @test @allocated(mean(ef)) === 0
            @test @allocated(var(ef)) === 0
            @test @allocated(basemeasure(ef, x)) === 0
            @test @allocated(sufficientstatistics(ef, x)) === 0
        end
    end
end

function run_test_fisherinformation_properties(distribution; test_properties_in_natural_space = true, test_properties_in_mean_space = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    if test_properties_in_natural_space
        F = getfisherinformation(NaturalParametersSpace(), T, conditioner)(η)

        @test issymmetric(F) || (LowerTriangular(F) ≈ (UpperTriangular(F)'))
        @test isposdef(F) || all(>(0), eigvals(F))
        @test size(F, 1) === size(F, 2)
        @test size(F, 1) === isqrt(length(F))
        @test (inv(fastcholesky(F)) * F ≈ Diagonal(ones(size(F, 1)))) rtol = 1e-2
    end

    if test_properties_in_mean_space
        θ = map(NaturalParametersSpace() => MeanParametersSpace(), T, η, conditioner)
        F = getfisherinformation(MeanParametersSpace(), T, conditioner)(θ)

        @test issymmetric(F) || (LowerTriangular(F) ≈ (UpperTriangular(F)'))
        @test isposdef(F) || all(>(0), eigvals(F))
        @test size(F, 1) === size(F, 2)
        @test size(F, 1) === isqrt(length(F))
        @test (inv(fastcholesky(F)) * F ≈ Diagonal(ones(size(F, 1)))) rtol = 1e-2
    end
end

function run_test_fisherinformation_against_hessian(distribution; assume_ours_faster = true, assume_no_allocations = true)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))

    @test fisherinformation(ef) ≈ ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), T, conditioner)(η), η)

    # Double check the `conditioner` free methods
    if isnothing(conditioner)
        @test fisherinformation(ef) ≈ ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), T)(η), η)
    end

    if assume_ours_faster
        @test @elapsed(fisherinformation(ef)) < (@elapsed(ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), T, conditioner)(η), η)))
    end

    if assume_no_allocations
        @test @allocated(fisherinformation(ef)) === 0
    end
end

function run_test_fisherinformation_against_jacobian(
    distribution;
    assume_no_allocations = true,
    mappings = (
        NaturalParametersSpace() => MeanParametersSpace(),
        MeanParametersSpace() => NaturalParametersSpace()
    )
)
    T = ExponentialFamily.exponential_family_typetag(distribution)

    ef = @inferred(convert(ExponentialFamilyDistribution, distribution))

    (η, conditioner) = (getnaturalparameters(ef), getconditioner(ef))
    θ = map(NaturalParametersSpace() => MeanParametersSpace(), T, η, conditioner)

    # Check natural to mean Jacobian based FI computation
    # So here we check that the fisher information matrices are identical with respect to `J`, which is the jacobian of the 
    # transformation. For example if we have a mapping T : M -> N, the fisher information matrices computed in M and N 
    # respectively must follow this relation `Fₘ = J' * Fₙ * J`
    for (M, N, parameters) in ((NaturalParametersSpace(), MeanParametersSpace(), η), (MeanParametersSpace(), NaturalParametersSpace(), θ))
        if (M => N) ∈ mappings
            mapping = getmapping(M => N, T)
            m = parameters
            n = mapping(m, conditioner)
            J = ForwardDiff.jacobian(Base.Fix2(mapping, conditioner), m)
            Fₘ = getfisherinformation(M, T, conditioner)(m)
            Fₙ = getfisherinformation(N, T, conditioner)(n)

            @test Fₘ ≈ (J' * Fₙ * J)

            # Check the default space
            if M === NaturalParametersSpace()
                # The `fisherinformation` uses the `NaturalParametersSpace` by default
                @test fisherinformation(ef) ≈ (J' * Fₙ * J)
            end

            # Double check the `conditioner` free methods
            if isnothing(conditioner)
                n = mapping(m)
                J = ForwardDiff.jacobian(mapping, m)
                Fₘ = getfisherinformation(M, T)(m)
                Fₙ = getfisherinformation(N, T)(n)

                @test Fₘ ≈ (J' * Fₙ * J)

                if M === NaturalParametersSpace()
                    @test fisherinformation(ef) ≈ (J' * Fₙ * J)
                end
            end

            if assume_no_allocations
                @test @allocated(getfisherinformation(M, T, conditioner)(m)) === 0
                @test @allocated(getfisherinformation(N, T, conditioner)(n)) === 0
            end
        end
    end
end

# This generic testing works only for the same distributions `D`
function test_generic_simple_exponentialfamily_product(
    left::Distribution,
    right::Distribution;
    strategies = (GenericProd(),),
    test_inplace_version = true,
    test_inplace_assume_no_allocations = true,
    test_preserve_type_prod_of_distribution = true,
    test_against_distributions_prod_if_possible = true
)
    Tl = ExponentialFamily.exponential_family_typetag(left)
    Tr = ExponentialFamily.exponential_family_typetag(right)

    @test Tl === Tr

    T = Tl

    efleft = @inferred(convert(ExponentialFamilyDistribution, left))
    efright = @inferred(convert(ExponentialFamilyDistribution, right))
    ηleft = @inferred(getnaturalparameters(efleft))
    ηright = @inferred(getnaturalparameters(efright))

    if (!isnothing(getconditioner(efleft)) || !isnothing(getconditioner(efright)))
        @test isapprox(getconditioner(efleft), getconditioner(efright))
    end

    prod_dist = prod(GenericProd(), left, right)

    # We check against the `prod_dist` only if we have the proper solution, and skip if the result is of type `ProductOf`
    if test_against_distributions_prod_if_possible && (prod_dist isa ProductOf || !(typeof(prod_dist) <: T))
        prod_dist = nothing
    end

    for strategy in strategies
        @test @inferred(prod(strategy, efleft, efright)) == ExponentialFamilyDistribution(T, ηleft + ηright, getconditioner(efleft))

        # Double check the `conditioner` free methods
        if isnothing(getconditioner(efleft)) && isnothing(getconditioner(efright))
            @test @inferred(prod(strategy, efleft, efright)) == ExponentialFamilyDistribution(T, ηleft + ηright)
        end

        # Check that the result is consistent with the `prod_dist`
        if !isnothing(prod_dist)
            @test convert(T, prod(strategy, efleft, efright)) ≈ prod_dist
        end
    end

    if test_inplace_version
        @test @inferred(prod!(similar(efleft), efleft, efright)) ==
              ExponentialFamilyDistribution(T, ηleft + ηright, getconditioner(efleft))

        if test_inplace_assume_no_allocations
            let _similar = similar(efleft)
                @test @allocated(prod!(_similar, efleft, efright)) === 0
            end
        end
    end

    if test_preserve_type_prod_of_distribution
        @test @inferred(prod(PreserveTypeProd(T), efleft, efright)) ≈
              prod(PreserveTypeProd(T), left, right)
    end

    return true
end
