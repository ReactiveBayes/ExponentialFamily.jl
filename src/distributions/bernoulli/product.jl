closed_prod_rule(::Type{<:Bernoulli}, ::Type{<:Bernoulli}) = ClosedProd()

function Base.prod(::ClosedProd, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > zero(norm) "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

closed_prod_rule(::Type{<:Bernoulli}, ::Type{<:Categorical}) = ClosedProd()

function Base.prod(::ClosedProd, left::Bernoulli, right::Categorical)
    p_left = probvec(left)
    p_right = probvec(right)

    max_length = max(length(p_left), length(p_right))

    p_new = zeros(promote_samplefloattype(p_left, p_right), max_length)

    e_left  = Iterators.flatten((p_left, Iterators.repeated(0, max(0, length(p_right) - length(p_left)))))
    e_right = Iterators.flatten((p_right, Iterators.repeated(0, max(0, length(p_left) - length(p_right)))))

    for (i, l, r) in zip(eachindex(p_new), e_left, e_right)
        @inbounds p_new[i] = l * r
    end

    return Categorical(normalize!(p_new, 1))
end