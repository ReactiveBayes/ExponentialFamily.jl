function tensordoubleindex(i, j, n)
    k = ceil(Int, i/n)
    l = ceil(Int, j/n)
    m = i - (k-1)*n
    n = j - (l-1)*n
    return k, l, m, n
end