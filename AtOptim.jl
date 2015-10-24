
"""
# module AtomsOptim

## Summary

Collection of methods to do geometry optimisation and related tasks. See
also the related `AtomsPrecon` module.

## List of methods


"""
module AtOptim

using AtomsInterface, AtPrecon

"""
# `function minimise!`

## Arguments
* `ftol` : terminate with success if maxforce <= ftol
* `precon=ExpPrecon(at, calc)` : preconditioner
* `direction=:steepestdescent` : the only one allowed right now,
* `lsmethod=:armijo` : the only one allowed right now,
* `alpha_init=1.0` : initial guess for alpha - reasonable for a good precon,
* `C1=0.1` : armijo condition (I don't believe in 1e-4),
* `min_alpha = 1e-8` : terminate with error is alpha < min_alpha,
* `maxnit=300` : terminate with error if nit > maxnit.
"""
function minimise!( at::AbstractAtoms, calc::AbstractCalculator, ftol;
                    precon=ExpPrecon(at, calc),
                    direction=:steepestdescent,
                    lsmethod=:armijo,
                    alpha_init=1.0,
                    C1=0.1,
                    min_alpha = 1e-8,
                    maxnit=300,
                    ftolP = ftol / 10 )
    
    @assert lsmethod==:armijo
    @assert direction==:steepestdescent
    @assert 0 1e-4 < C1 < 0.5
    
    # first round of energy and force evaluations
    E = potential_energy(at, calc)
    frc = forces(at, calc)
    
    for n = 1:maxnit
        # check for termination
        if maxforce(frc) <= ftol
            return at, E, frc
        end
        
        # precondition the force
        frcP = frc / precon
        
        # do an extrapolation to compute a good initial step-length guess
        if n == 1
            alpha = alpha_init
        else
            Delta_g_sq = vecdot(frc-old_frc, frcP-old_frcP)
            alpha = vecdot(step, frc-old_frc) / Delta_g_sq
            # reset if alpha is ridiculously small
            alpha < min_alpha ? alpha = alpha_init
        end
        
        # call the linesearch
        #   only at is modified in the linesearch, but not X!
        alpha, new_E, new_frc = ls_armijo(at, calc, E, frc, frcP,
                                          alpha, alpha_min, C1)

        # update everything
        step = alpha * frcP
        old_frc, old_frcP, old_E, E = frc, frcP, E, E_new
        isempty(new_frc) ? frc = forces(at, calc) : frc = new_frc
    end
    
end



"""
`ls_armijo!(at, calc, E0, frc0, dir, alpha0, alpha_min, C1)`: 

Backtracking Armijo linesearch, using a quadratic interpolation step
to improve the quality of backtracking.

Modifies only `at` (and implicitly `calc`)
"""
function ls_armijo!(at, calc, E0, frc0, dir, alpha0, alpha_min, C1)

    alpha = alpha0
    dir_dot_frc0 = vecdot(frc0, dir)
    X = positions(at)
    
    while alpha >= alpha_min
        # compute energy at current trial
        set_positions!(at, X + alpha0 * p)
        E1 = energy(at, calc)
        # check armijo condition
        if f1 <= f0 - C1 * alpha * dir_dot_frc0
            return alpha, E1, Float64[]
        end
        # compute an updated alpha (make sure there is not too much decrease)
        at = - (E0 * alpha^2) / (2 * (E1 - E0 - dir_dot_frc0 * a1))
        alpha = max(at, alpha / 10.0)
    end
    
    error("ls_armijo! : alpha < alpha_min ")
end
