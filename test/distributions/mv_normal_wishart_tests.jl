
@testitem "MvNormalWishart: common" begin
    include("distributions_setuptests.jl")

    m = rand(2)
    dist = MvNormalWishart(m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
    @test params(dist) == (m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
    @test dof(dist) == 3.0
    @test invscatter(dist) == [1.0 0.0; 0.0 1.0]
    @test scale(dist) == 0.1
    @test locationdim(dist) == 2
end

@testitem "MvNormalWishart: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for dim in (3,), invS in rand(Wishart(10, Array(Eye(dim))), 4)
        ν = dim + 2
        @testset let (d = MvNormalWishart(rand(dim), invS, rand(), ν))
            ef = test_exponentialfamily_interface(
                d;
                option_assume_no_allocations = false,
                test_basic_functions = false,
                test_fisherinformation_against_hessian = false,
                test_fisherinformation_against_jacobian = false
            )

            run_test_basic_functions(d; assume_no_allocations = false, test_samples_logpdf = false)
        end
    end
end

@testitem "MvNormalWishart: prod" begin
    include("distributions_setuptests.jl")

    for j in 2:2, κ in 1:2
        m1 = rand(j)
        m2 = rand(j)
        Ψ1 = m1 * m1' + I
        Ψ2 = m2 * m2' + I
        dist1 = MvNormalWishart(m1, Ψ1, κ + rand(), rand() + 4)
        dist2 = MvNormalWishart(m2, Ψ2, κ + rand(), rand() + 4)
        ef1 = convert(ExponentialFamilyDistribution, dist1)
        ef2 = convert(ExponentialFamilyDistribution, dist2)
        @test prod(PreserveTypeProd(Distribution), dist1, dist2) ≈ convert(Distribution, prod(ClosedProd(), ef1, ef2))
    end
end

@testitem "MvNormalWishart: prod with ExponentialFamilyDistribution{MvNormalWishart}" begin
    include("distributions_setuptests.jl")

    for Sleft in rand(Wishart(10, Array(Eye(2))), 2), Sright in rand(Wishart(10, Array(Eye(2))), 2), νright in (6, 7), νleft in (4, 5)
        @testset let (left, right) = (MvNormalWishart(rand(2), Sleft, rand(), νleft), MvNormalWishart(rand(2), Sright, rand(), νright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (PreserveTypeProd(ExponentialFamilyDistribution{MvNormalWishart}), GenericProd())
            )
        end
    end
end
