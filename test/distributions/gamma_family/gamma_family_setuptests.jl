
include("../distributions_setuptests.jl")

import ExponentialFamily: xtlog

function compare_basic_statistics(left, right)
    @test mean(left) ≈ mean(right)
    @test var(left) ≈ var(right)
    @test cov(left) ≈ cov(right)
    @test shape(left) ≈ shape(right)
    @test scale(left) ≈ scale(right)
    @test rate(left) ≈ rate(right)
    @test entropy(left) ≈ entropy(right)
    @test pdf(left, 1.0) ≈ pdf(right, 1.0)
    @test pdf(left, 10.0) ≈ pdf(right, 10.0)
    @test logpdf(left, 1.0) ≈ logpdf(right, 1.0)
    @test logpdf(left, 10.0) ≈ logpdf(right, 10.0)

    @test mean(log, left) ≈ mean(log, right)
    @test mean(loggamma, left) ≈ mean(loggamma, right)
    @test mean(xtlog, left) ≈ mean(xtlog, right)

    return true
end