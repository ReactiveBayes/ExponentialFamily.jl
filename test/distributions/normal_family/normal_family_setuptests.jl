include("../distributions_setuptests.jl")

import ExponentialFamily: dot3arg

# We need this extra function to ensure better derivatives with AD, it is slower than our implementation
# but is more AD friendly
function getlogpartitionfortest(::NaturalParametersSpace, ::Type{MvNormalMeanCovariance})
    return (η) -> begin
        weightedmean, minushalfprecision = unpack_parameters(MvNormalMeanCovariance, η)
        return (dot3arg(weightedmean, inv(-minushalfprecision), weightedmean) / 2 - logdet(-2 * minushalfprecision)) / 2
    end
end

function gaussianlpdffortest(params, x)
    k = length(x)
    μ, Σ = params[1:k], reshape(params[k+1:end], k, k)
    coef = (2π)^(-k / 2) * det(Σ)^(-1 / 2)
    exponent = -0.5 * (x - μ)' * inv(Σ) * (x - μ)
    return log(coef * exp(exponent))
end

function check_basic_statistics(left::UnivariateNormalDistributionsFamily, right::UnivariateNormalDistributionsFamily)
    @test mean(left) ≈ mean(right)
    @test median(left) ≈ median(right)
    @test mode(left) ≈ mode(right)
    @test var(left) ≈ var(right)
    @test std(left) ≈ std(right)
    @test entropy(left) ≈ entropy(right)

    for value in (1.0, -1.0, 0.0, mean(left), mean(right), rand())
        @test pdf(left, value) ≈ pdf(right, value)
        @test logpdf(left, value) ≈ logpdf(right, value)
        @test all(
            ForwardDiff.gradient((x) -> logpdf(left, x[1]), [value]) .≈
            ForwardDiff.gradient((x) -> logpdf(right, x[1]), [value])
        )
        @test all(
            ForwardDiff.hessian((x) -> logpdf(left, x[1]), [value]) .≈
            ForwardDiff.hessian((x) -> logpdf(right, x[1]), [value])
        )
    end

    # `Normal` is not defining some of these methods and we don't want to define them either, because of the type piracy
    if !(left isa Normal || right isa Normal)
        @test cov(left) ≈ cov(right)
        @test invcov(left) ≈ invcov(right)
        @test weightedmean(left) ≈ weightedmean(right)
        @test precision(left) ≈ precision(right)
        @test all(mean_cov(left) .≈ mean_cov(right))
        @test all(mean_invcov(left) .≈ mean_invcov(right))
        @test all(mean_precision(left) .≈ mean_precision(right))
        @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
        @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
        @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
    end
end

function check_basic_statistics(left::MultivariateNormalDistributionsFamily, right::MultivariateNormalDistributionsFamily)
    @test mean(left) ≈ mean(right)
    @test mode(left) ≈ mode(right)
    @test var(left) ≈ var(right)
    @test cov(left) ≈ cov(right)
    @test logdetcov(left) ≈ logdetcov(right)
    @test length(left) === length(right)
    @test size(left) === size(right)
    @test entropy(left) ≈ entropy(right)

    dims = length(mean(left))

    for value in (
        fill(1.0, dims),
        fill(-1.0, dims),
        fill(0.1, dims),
        mean(left),
        mean(right),
        rand(dims)
    )
        @test pdf(left, value) ≈ pdf(right, value)
        @test logpdf(left, value) ≈ logpdf(right, value)
        @test all(
            isapprox.(
                ForwardDiff.gradient((x) -> logpdf(left, x), value),
                ForwardDiff.gradient((x) -> logpdf(right, x), value),
                atol = 1e-12
            )
        )
        if !all(
            isapprox.(
                ForwardDiff.hessian((x) -> logpdf(left, x), value),
                ForwardDiff.hessian((x) -> logpdf(right, x), value),
                atol = 1e-12
            )
        )
            error(left, right)
        end
    end

    # `MvNormal` is not defining some of these methods and we don't want to define them either, because of the type piracy
    if !(left isa MvNormal || right isa MvNormal)
        @test ndims(left) === ndims(right)
        @test invcov(left) ≈ invcov(right)
        @test weightedmean(left) ≈ weightedmean(right)
        @test precision(left) ≈ precision(right)
        @test all(mean_cov(left) .≈ mean_cov(right))
        @test all(mean_invcov(left) .≈ mean_invcov(right))
        @test all(mean_precision(left) .≈ mean_precision(right))
        @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
        @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
        @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
    end
end