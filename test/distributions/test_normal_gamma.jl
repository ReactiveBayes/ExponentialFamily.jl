function normal_gamma_pdf(x, τ, μ, λ, α, β)
    return pdf(NormalMeanPrecision(μ, λ * τ), x) * pdf(GammaShapeRate(α, β), τ)
end
