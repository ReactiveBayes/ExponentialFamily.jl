@testitem "Multinomial: probvec" begin
    include("distributions_setuptests.jl")

    @test probvec(Multinomial(5, [1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
    @test probvec(Multinomial(3, [0.2, 0.2, 0.4, 0.1, 0.1])) == [0.2, 0.2, 0.4, 0.1, 0.1]
    @test probvec(Multinomial(2, [0.5, 0.5])) == [0.5, 0.5]
end

@testitem "Multinomial: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Multinomial)
    @test_throws MethodError vague(Multinomial, 4)

    vague_dist1 = vague(Multinomial, 5, 4)
    @test typeof(vague_dist1) <: Multinomial
    @test probvec(vague_dist1) == [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    vague_dist2 = vague(Multinomial, 3, 5)
    @test typeof(vague_dist2) <: Multinomial
    @test probvec(vague_dist2) == [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
end

@testitem "Multinomial: test_EF_interface" begin
    include("distributions_setuptests.jl")

    using StableRNGs

    rng = StableRNG(42)

    for n in 2:6
        for trials in 2:10
            @testset let d = Multinomial(trials, normalize!(rand(rng, n), 1))
                test_exponentialfamily_interface(d; option_assume_no_allocations = false,
                    test_fisherinformation_properties = false,
                    test_fisherinformation_against_jacobian = false)
            end
        end
    end
end

@testitem "Product of Multinomial distributions" begin
    include("distributions_setuptests.jl")

    using StableRNGs
    using Distributions: Uniform

    rng = StableRNG(42)

    @testset "prod" begin
        for n in 4:6
            pleft = rand(rng, n)
            pleft = pleft ./ sum(pleft)
            pright = rand(rng, n)
            pright = pright ./ sum(pright)
            left = Multinomial(n, pleft)
            right = Multinomial(n, pright)
            efleft = convert(ExponentialFamilyDistribution, left)
            efright = convert(ExponentialFamilyDistribution, right)
            prod_ef = prod(PreserveTypeProd(ExponentialFamilyDistribution), efleft, efright)
            d = vague(Multinomial, n, n)
            sample_space = unique(rand(rng, d, 4000), dims = 2)
            sample_space = [sample_space[:, i] for i in 1:size(sample_space, 2)]

            # Test normalization for the new interface
            hist_sumef(x) =
                getbasemeasure(prod_ef)(x) * exp(
                    getnaturalparameters(prod_ef)' * first(sufficientstatistics(prod_ef, x)) -
                    logpartition(prod_ef, getnaturalparameters(prod_ef))
                )
            @test sum(hist_sumef(x_sample) for x_sample in sample_space) ≈ 1.0 rtol = 1e-3

            # Test basemeasure and sufficient statistics
            sample_x = rand(d, 5)
            for xi in sample_x
                @test getbasemeasure(prod_ef)(xi) ≈ (factorial(n) / prod(@.factorial(xi)))^2 rtol = 1e-10
                @test sufficientstatistics(prod_ef, xi) == (xi,)
            end
        end

        # Test error cases for mismatched conditioners
        @test_throws AssertionError prod(
            PreserveTypeProd(ExponentialFamilyDistribution),
            convert(ExponentialFamilyDistribution, Multinomial(4, [0.2, 0.4, 0.4])),
            convert(ExponentialFamilyDistribution, Multinomial(5, [0.1, 0.3, 0.6]))
        )
        @test_throws AssertionError prod(
            PreserveTypeProd(ExponentialFamilyDistribution),
            convert(ExponentialFamilyDistribution, Multinomial(4, [0.2, 0.4, 0.4])),
            convert(ExponentialFamilyDistribution, Multinomial(3, [0.1, 0.3, 0.6]))
        )
    end
end