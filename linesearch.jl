module linesearch

export golden

function golden(f, lo, hi)
    if lo > hi
        error("golden(): lo ($lo) > hi ($hi)")
    end
    slo = sign(f(lo))
    shi = sign(f(hi))
    if slo == shi
        error("golden(): sign(f(lo)) == sign(f(hi)) == $(slo)")
    end
    max_iter = 100
    for i=1:max_iter
        mi=0.5*(lo+hi)
        if sign(f(mi)) == slo
            lo=mi;
        else
            hi=mi
        end
        if (hi-lo) < 100*eps(hi)
            return mi
        end
    end
    error("golden(): failed to find zero in $max_iter steps. Current bracket: [$lo,$hi]")
end

end