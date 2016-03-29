module md
global f_old

export md_step



function md_langevin_step_baoab(q, p, f, m, dt, kT, gamma)
    
    global f_old
    #baoab discretisation of Langevin dynamics
    
    if !isdefined(f_old)
        f_old = f
    end
    
    p = p + (dt/2)*f_old
    q = q + (dt/2)*p./m
    p = p * exp(-dt * gamma) + sqrt(kT*(1-exp(-2*dt*gamma))*m).*randn(size(p))
    q = q + (dt/2)*p./m
    p = p + (dt/2)*f
    f_old = f
end

#alias
md_step = md_langevin_step_baoab

end