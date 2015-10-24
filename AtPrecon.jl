
"""
# module AtPrecon

## Summary

Implements preconditioners for molecular simulation
    
## List of Types

* `PairPrecon` : our standard preconditioner; normally created using
    -  ExpPrecon

## Typical usage

Construct a preconditioner:
```
# construct at::ASEAtoms, calc
precon = ExpPrecon(at, calc)
```
`precon` can be passed  to an optimiser. In the optimiser
```
u * precon
```
multiplies a 3 x N displacement vector with the preconditioner, while
```
frc / precon
```
computes the preconditioned force vector from a 3 x N raw force vector
"""
module AtPrecon


using AtomsInterface, MatSciPy, ASE, Potentials

export init!, update!





"""
# type PairPrecon <: Preconditioner

c_ij = μ * ϕ(r_ij)
c_ii = - ∑_{j ≠ i} c_ij + μ Cstab

where ϕ is an arbitrary pair potential.

Update will be performed when there is movement of at least
    `pupdate * r0`
"""
type PairPrecon{T <: PairPotential} <: Preconditioner
    pp::T           # pair potential ϕ
    mu::Float64     # energy-scale
    r0::Float64     # approximate NN distance
    Rc::Float64     # cut-off
    cstab::Float64  # stabilisation constant
    pupdate::Float64  # after how much movement does the precon need updating
                      # given as a factor of r0
    # solver::Symbol
    arrays::Dict
end


"""
# type ExpPrecon

`PairPrecon` specification with ϕ = exp( - A (r/r0 - 1) )

Constructors: 

* `ExpPrecon(; A=3.0, r0=2.5 mu=1.0, Rc=4.5, cstab=0.0 )` 
* `ExpPrecon(at::AbstractAtoms, calc::AbstractCalculator;
          mask = ones(3), kwargs... )` : same as first + initialises
"""
typealias ExpPrecon PairPrecon{SimpleExponential}


ExpPrecon(; A=3.0, r0=2.5
          mu=1.0, Rc=4.5, cstab=0.0, rupdate=0.2 ) =
              PairPrecon( SimpleExponential(1.0, -A, r0),
                          mu, r0, Rc, cstab, pupdate, Dict() )

ExpPrecon(at::AbstractAtoms, calc::AbstractCalculator;
          mask = ones(3), kwargs... ) =
              init!( ExpPrecon( kwargs... ), at, calc; mask=mask)


"pure matrix assembly component"
function assemble!(precon, at)
    i, j, r = MatSciPy.neighbour_list(at, "ijd", precon.Rc)
    c = precon.mu * precon.pp(r)
    P = sparse(i, j, r, (length(at), length(at)))
    P += diagm( - sum(P, 1) + precon.mu * precon.cstab )
    return P
end


"""compute a good mu

  -< frc(X+h V) - frc(X), V > ~ mu <P V, V>
  where V_i(x) = ∑_{j=1}^3 Mij sin( xj * 2*pi/Li )
    X (0, Li) is the computational cell and Mij is a mask-matrix

  hfd: finite difference step (relative to r0)
"""
function estimate_mu_factor(at, calc, P, mask; hfd = 1e-3)
    X = positions(at)
    Linv = 2*pi ./ diag(cell(at))
    if length(size(mask)) == 1
        mask = diag(mask)
    end
    # compute the two force vectors and the RHS <Pv, v>
    V = hfd * precon.r0 * mask * sin(L .* X)
    frc = forces(at, calc)
    set_positions!(at, X + V)
    frcV = forces(at, calc)
    # return the good mu
    new_mu = vecdot(frc - frcV, V) / vecdot(V * P, V)
    if mu <= 1e-10
        error("""mu < 1e-10 : this probably means the test configuration is
              (long-wavelength-)unstable; either give me a better configuration 
              for  initialising the preconditioner, or construct a good one by
              hand. A lattice ground state is normally a safe choice.
              """)
    end
    return new_mu
end



"""
`init!(precon::PairPrecon, at::AbstractAtoms, calc::AbstractCalculator;
               mask = ones(3) )`

Initialises a preconditioner based on a concrete atoms object and a 
concrete calculator. At the moment only the μ parameter is fitted.

## Non-obvious Parameters

* `mask` : should be either a 3-dimensional vector of bools which determine in 
           which direction the test function is non-constant; or a general
           3 x 3 matrix

"""
function init!(precon::PairPrecon, at::AbstractAtoms, calc::AbstractCalculator;
               mask = ones(3) )
    # make sure that at has a cubic computational cell
    assert_cubic(at)
    # initial assembly (no stabilisation!)
    cstab = precon.cstab; precon.cstab = 0.0
    P = assemble!(precon, at)
    precon.cstab = cstab
    # do the mu - test, and adjust the mu parameter
    precon.mu *= estimate_mu_factor(at, calc, P, mask)
    # assemble again, by calling update!
    update!(precon, at)
    return precon
end


"check whether the preconditioner needs updating"
function need_update(precon, at)
    # update if the preconditioner has never been updated before
    if get_array(precon, :X) == nothing
        return true
    # also update there is sufficient movement.
    #    NOTE: this is very crude and should be able to potentially take
    #          periodicity into account  (TODO)    
    elseif ( max(sumabs2(get_array(precon, :X) - positions(at), 1), 2) >
             (precon.pupdate*precon.r0)^2 )
        return true
    end
    return false
end



"""
`update!(precon::PairPrecon, at::ASEAtoms)`

check whether atoms have moved a lot and if they have, update the 
matrix as well as the solver.
"""
function update!(precon::PairPrecon, at::ASEAtoms)
    assert_cubic(at)
    if need_update(precon, at)
        P = assemble!(precon, at)
        set_array(precon, P, :P)
        set_array(precon, lufact(P), :solver)
    end
end


import Base./, Base.*, Base.dot


# multiply with precon
*(frc::Matrix, precon::PairPrecon) = frc * get_array(precon, :P)
# precon-dot product
dot(frc::Matrix, precon::PairPrecon) = vecdot(frc * precon, frc)
# precon-solver for N x 1 array
\(precon::PairPrecon, v::AbstractVector) = get_array(precon, :solver) \ v
# precon-solver for 3 x N force arrays
function /(frc::Matrix, precon::PairPrecon)
    pfrc = zeros(size(frc))
    for n = 1:size(frc, 1)
        pfrc[n,:] = get_array(precon, :solver) \ slice(frc, n, :)
    end
end


end
