
"""
# module AtomsOptim

## Summary

Collection of methods to do geometry optimisation and related tasks. See
also the related `AtomsPrecon` module.

## List of exported methods

* `minimise!` : 

"""
module AtOptim

using AtomsInterface, AtPrecon

export minimise!, OptimHistory


type OptimHistory
    num_E
    num_frc
    time
end

OptimHistory() = OptimHistory(Float64[], Float64[], Float64[])


"""
# `function minimise!`

Right now this implements the preconditioned steepest descent method
with BB step-size guess and a quadratic interpolation backtracking
Armijo linesearch. 

TODO: extend to include CG, LBFGS

## Arguments
* `ftol` : terminate with success if maxforce <= ftol
* `precon=ExpPrecon(at, calc)` : preconditioner
* `direction=:steepestdescent` : the only one allowed right now,
* `lsmethod=:armijo` : the only one allowed right now,
* `alpha_init=1.0` : initial guess for alpha - reasonable for a good precon,
* `C1=0.1` : armijo condition (I don't believe in 1e-4),
* `min_alpha = 1e-8` : terminate with error is alpha < min_alpha,
* `maxnit=300` : terminate with error if nit > maxnit.
* `disp=0` : 0 = no display, 1 = at termination, 2 = at each iteration,
             3 = full debugging info
"""
function minimise!( at::AbstractAtoms, calc::AbstractCalculator, ftol;
                    precon=ExpPrecon(at, calc),
                    direction=:steepestdescent,
                    lsmethod=:armijo,
                    alpha_init=1.0,
                    C1=0.1,
                    min_alpha = 1e-8,
                    maxnit = 300,
                    ftolP = ftol / 10,
                    disp = 0 )
    
    # @assert lsmethod==:armijo
    # @assert direction==:steepestdescent
    # @assert 1e-4 < C1 < 0.5

    # choose the extrapolation mechanism: hard-coded
    extrapolate_alpha = extrap_bb1
    # choose the linesearch mechanims
    linesearch! = ls_armijo!
    
    # first round of energy and force evaluations
    E = potential_energy(at, calc)
    frc = forces(at, calc)
    # on the first linesearch:
    alpha = alpha_init
    alpha_hist = Float64[]

    num_E = 1
    if disp ≥ 2
        @printf("-------|---------------------------------------------------------\n")
        @printf(" nit   |   ΔE        |∇E|∞     |Δx|∞      α        #E \n")
        @printf("-------|---------------------------------------------------------\n")
    end

    # main iteration
    old_frc = old_frcP = step = old_E = 0.0
    for n = 1:maxnit
        fres = maxforce(frc)
        if disp >= 2
            @printf(" %5d | %4.2e  %4.2e  %4.2e  %4.2e  %5d \n",
                    n, E - old_E, fres, vecnorm(step, Inf), alpha, num_E)
        end
        
        # check for termination
        if fres <= ftol
            @printf("-------|---------------------------------------------------------\n")
            return at, E, frc
        end
        
        # precondition the force
        frcP = frc / precon
        
        # do an extrapolation to compute a good initial step-length guess
        # then also ensure that the change is not too crazy.
        if n > 1
            alpha = extrapolate_alpha(E, frc, frcP,
                                      old_E, old_frc, old_frcP, step,
                                      alpha_hist)
        end
        
        # call the linesearch
        #   only at is modified in the linesearch, but not X!
        alpha, new_E, new_frc, num_E_plus =
            linesearch!(at, calc, E, frc, frcP, alpha, min_alpha, C1)
        # remember the history
        push!(alpha_hist, alpha)
        num_E += num_E_plus

        # update everything
        step = alpha * frcP
        old_frc, old_frcP, old_E, E = frc, frcP, E, new_E
        if size(new_frc) == size(frc)
            frc = new_frc
        else
            frc = forces(at, calc)
        end
        update!(precon, at)
    end
end


"""
`extrap_bb1`

First type of Barzilai-Borwein step selection
"""
function extrap_bb1(E, frc, frcP,
                    old_E, old_frc, old_frcP, step, alpha_hist)
    Delta_g_sq = vecdot(frc - old_frc, frcP - old_frcP)
    alpha = vecdot(step, old_frc - frc) / Delta_g_sq
    alpha = min( max( alpha, alpha_hist[end] / 10 ), alpha_hist[end] * 10 )
    return alpha
end



using TestAtoms

"""
`ls_armijo!(at, calc, E0, frc0, dir, alpha0, alpha_min, C1)`: 

Backtracking Armijo linesearch, using a quadratic interpolation step
to improve the quality of backtracking.

Modifies only `at` (and implicitly `calc`)
"""
function ls_armijo!(at, calc, E0, frc0, dir,
                    alpha0, alpha_min, C1)

    dir_dot_frc0 = vecdot(frc0, dir)
    X0 = positions(at)
    num_E = 0
    alpha = alpha0
    
    
    
    while alpha >= alpha_min
        # compute energy at current trial
        # E1 ~ E0 + alpha * < dE0, dir > ~ E0 - alpha * <frc0, dir>
        set_positions!(at, X0 + alpha * dir)
        E1 = potential_energy(at, calc)
        num_E += 1
        # check armijo condition
        if E1 <= E0 - C1 * alpha * dir_dot_frc0
            return alpha, E1, [], num_E
        end
        # compute an updated alpha (make sure there is not too much decrease)
            # at = -  ((self.phi_prime_start * a1) /
            #     (2*((phi_a1 - self.func_start)/a1 - self.phi_prime_start)))
        alpha_t = (dir_dot_frc0 * alpha) / (2 * ((E1 - E0)/alpha + dir_dot_frc0))
        alpha = max(alpha_t, alpha / 10.0)
    end
    
    error("ls_armijo! : alpha < alpha_min ")
end



end
