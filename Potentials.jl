
"""
## module Potentials

### Summary

This module implements some basic interatomic potentials in pure Julia.The
implementation is done in such a way that they can be used either in "raw" form
or within abstract frameworks.

### Types



"""
module Potentials

using Prototypes, AtomsInterface
importall AtomsInterface

export PairPotential, SimpleFunction, SitePotential, AbstractCutoff
export LennardJonesPotential, SWCutoff, ShiftCutoff, EAMPotential
export SimpleExponential, GuptaEmbed, GuptaPotential
export evaluate, evaluate_d, evaluate_dd, @D, @DD, @GRAD, grad
export cutoff, MorsePotential, ZeroSitePotential, ZeroPairPotential
export evaluate_d!, grad!, evaluate!

export @D2

# some constants used with the SWCutoff
const _eps_ = eps()
const _eps2_ = 1e-2

#########################################################
###         Pair Potentials                          
#########################################################

abstract SimpleFunction
abstract PairPotential <: SimpleFunction
abstract SitePotential


"""`evaluate(pp::SimpleFunction, r)`: evaluate a scalar potential at `r`; 
typically a pair potential, where `r` may be a scalar or an array (of scalars)
"""
@protofun evaluate(pp::SimpleFunction, r)
"`evaluate_d(pp::PairPotential, r)`: first derivative of pair potential"
@protofun evaluate_d(pp::SimpleFunction, r)
"`evaluate_d2(pp::PairPotential, r)`: second derivative of pair potential"
@protofun evaluate_dd(pp::SimpleFunction, r)

# A function may also depend on a vector R. Since neighbourlists often
# provide the scalar r = norm(R), we can also evaluate SimpleFunctions
# using the form
#    evaluate(p::SimpleFunction, r, R)
#
# But if it is a PairPotential, then we should assume - by default - that
# it depends only on r, and hence implement the following defaults
evaluate(p::PairPotential, r, R) = evaluate(p, r)
evaluate_d(p::PairPotential, r, R) = evaluate_d(p, r)
evaluate_dd(p::PairPotential, r, R) = evaluate_dd(p, r)

# prototype-in-place operations
"""
# evaluate_d!(p, r, R, dP)

in-place variant of `evaluate_d`
"""
function evaluate_d!(p, r, R, dP)
    d = evaluate_d(p, r, R)
    for ind in eachindex(d)
        dP[ind] = d[ind]
    end
end

"""# evaluate!(p, r, R, P)

in-place variant of `evaluate`.
"""
function evaluate!(p, r, R, P)
    P[:] = evaluate(p, r, R)
    return P
end

"""
# evaluate_d!(p, r, R, dP)

in-place variant of `grad`
"""
function grad!(p, r, R, G)
    g = grad(p, r, R)
    for ind in eachindex(g)
        G[ind] = g[ind]
    end
end


# but for grad, this doesn't work. the default here are given by
"""`grad(pp::PairPotential, R)`: if `R` is a 3 x N matrix (or 3 -vector) then
`grad` returns a 3x N array `G` whose columns are the gradient of the 
pair potential as a function of |R[:,i]|.

if `r = sqrt(sumabs2(R, 1))` is already available, then calling 
`grad(pp, r, R)` may be more efficient.

see also `@GRAD`.
"""
@inline grad(pp::PairPotential, R) = grad(pp, sqrt(sumabs2(R, 1)), R)
@inline grad(pp::PairPotential, r, R) = R .* (evaluate_d(pp, r) ./ r)'
@inline grad(pp::PairPotential, r, R) = R .* (evaluate_d(pp, r, R) ./ r)'


### The next block of code is an attempt to use call-overloading and macros to
### allow the following syntax:
###   pp  ... a pair potential
###   r ... a length or array of lengths
###   ϕ = pp(r)
###   ϕ' = @D pp(r)
###   ϕ'' = @DD pp(r)
###
# create an alias to allow the potential to be called directly
#  make it an inline to ensure that nothing is lost here!
#
# BETTER: call(::Type{Val{:D}}, pp::SimpleFunction, varargs...) = evaluate_d(pp, varargs...)
#
import Base.call
typealias FunUn Union{SimpleFunction, SitePotential}
@inline call(pp::FunUn, varargs...) = evaluate(pp, varargs...)
@inline call(pp::FunUn, ::Type{Val{:D}}, varargs...) =
    evaluate_d(pp, varargs...)
@inline call(pp::FunUn, ::Type{Val{:DD}}, varargs...) =
    evaluate_dd(pp, varargs...)
@inline call(pp::FunUn, ::Type{Val{:GRAD}}, varargs...) =
    grad(pp, varargs...)


# @inline call(pp::SimpleFunction, r) = evaluate(pp, r)
# @inline call(pp::SimpleFunction, r, ::Type{Val{:D}}) = evaluate_d(pp, r)
# @inline call(pp::SimpleFunction, r, ::Type{Val{:DD}}) = evaluate_dd(pp, r)

# @inline call(pp::PairPotential, R, ::Type{Val{:GRAD}}) = grad(pp, R)

# @inline call(pp::SimpleFunction, r, R) = evaluate(pp, r, R)
# @inline call(pp::PairPotential, r, R, ::Type{Val{:GRAD}}) = grad(pp, r, R)

# # now the same for site potentials;
# @inline call(p::SitePotential, R) = evaluate(p, R)
# @inline call(p::SitePotential, R, ::Type{Val{:D}}) = evaluate_d(p, R)
# @inline call(p::SitePotential, R, ::Type{Val{:GRAD}}) = grad(p, R)
# @inline call(p::SitePotential, r, R) = evaluate(p, R)
# @inline call(p::SitePotential, r, R, ::Type{Val{:D}}) = evaluate_d(p, r, R)
# @inline call(p::SitePotential, r, R, ::Type{Val{:GRAD}}) = grad(p, r, R)




# next create macros that translate
"""`@D`

This macro can be used to evaluate the derivative of a potential, e.g., 
of a pair potential. For example, to compute the Lennard-Jones potential,
```
lj = LennardJonesPotential()
r = 1.0 + rand(10)
ϕ = lj(r)
ϕ' = @D lj(r)
```

see also `@DD`.
"""
macro D(fsig::Expr)
    @assert fsig.head == :call
    insert!(fsig.args, 2, Val{:D})
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    return fsig
end

"`@DD` : analogous to `@D`"
macro DD(fsig::Expr)
    @assert fsig.head == :call 
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    insert!(fsig.args, 2, Val{:DD})
    return fsig
end

"""`@GRAD` : If `p` is a `PairPotential`, and 
R a d x N array, then  `@GRAD p(R)` returns an array G
"""
macro GRAD(fsig::Expr)
    @assert fsig.head == :call 
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    insert!(fsig.args, 2, Val{:GRAD})
    return fsig
end


#########################################################
###         Zero Functions
"`ZeroPairPotential`: pair potential V(r) = 0.0"
type ZeroPairPotential <: PairPotential end
evaluate(p::ZeroPairPotential, r) = zeros(size(r))
evaluate_d(p::ZeroPairPotential, r) = zeros(size(r))
grad(p::ZeroPairPotential, r, R) = zeros(size(R))
cutoff(p::ZeroPairPotential) = 0.0

"`ZeroSitePotential`: Site potential V(R) = 0.0"
type ZeroSitePotential <: SitePotential end
evaluate(p::ZeroSitePotential, R) = 0.0
evaluate_d(p::ZeroSitePotential, R) = zeros(size(R))
evaluate(p::ZeroSitePotential, r, R) = 0.0
evaluate_d(p::ZeroSitePotential, r, R) = zeros(size(R,1),1,1,size(R,2))
grad(p::ZeroSitePotential, R) = zeros(3,1,1,size(R,1))
grad(p::ZeroSitePotential, r, R) = zeros(3,1,1,size(R,2))
cutoff(p::ZeroSitePotential) = 0.0


#########################################################
### Cut-off potentials

""" An `AbstractCutoff` is a type that, when evaluated will return the cut-off
variant of that potential. 

All `Cutoff <: AbstractCutoff` must have a constructor of the form
```
    Cutoff(pp::PairPotential, args...)
```
"""
abstract AbstractCutoff <: PairPotential

# abstract implementation for getting the cut-off radius, assuming that
# it is stored as p.Rc
cutoff(p::AbstractCutoff) = p.Rc


########################

"""`cutsw(r, Rc, Lc)`

Implementation of the C^∞ Stillinger-Weber type cut-off potential
    1.0 / ( 1.0 + exp( lc / (Rc-r) ) )

`d_cutinf` implements the first derivative
"""
@inline cutsw(r, Rc, Lc) =
    1.0 ./ ( 1.0 + exp( Lc ./ ( max(Rc-r, 0.0) + _eps_ ) ) )

"derivative of `cutsw`"
@inline function cutsw_d(r, Rc, Lc)
    t = 1 ./ ( max(Rc-r, 0.0) + _eps_ )   # a numerically stable (Rc-r)^{-1}
    e = 1.0 ./ (1.0 + exp(Lc * t))                     # compute exponential only once
    return - Lc * (1.0 - e) .* e .* t.^2    
end


# function cutoff_NRL(r::Float64, Rc, lc)
#     fcut = r > Rc ? 0.0 : 1.0 / ( 1.0 + exp( (r-Rc)/lc + 5.0 ) )
#     return fcut
# end


"""
`type SWCutoff`: SW type cut-off potential with C^∞ regularity
     1.0 / ( 1.0 + exp( Lc / (Rc-r) ) )

This is not very optimised: one could speed up `evaluate_d` significantly
by avoiding multiple evaluations.
"""
type SWCutoff <: AbstractCutoff
    pp::PairPotential
    Rc::Float64
    Lc::Float64
end
@inline evaluate(p::SWCutoff, r) =
    p.pp(r) .* cutsw(r, p.Rc, p.Lc)
@inline evaluate_d(p::SWCutoff, r) =
    p.pp(r) .* cutsw_d(r, p.Rc, p.Lc) + (@D p.pp(r)) .* cutsw(r, p.Rc, p.Lc)
# `cutoff` is automatically implemented because of field Rc


########################

"""
`ShiftCutoff` : takes the pair-potential and shifts and truncates it
    f_cut(r) = (f(r) - f(rcut)) .* (r <= rcut)

Default constructor is
```
    ShiftCutoff(Rc, pp)
```
"""
type ShiftCutoff <: AbstractCutoff
    pp::PairPotential
    Rc::Float64
    Jc::Float64
end
ShiftCutoff(pp, Rc) = ShiftCutoff(pp, Rc, pp(Rc))
@inline evaluate(p::ShiftCutoff, r) = (p.pp(r) - p.Jc) .* (r .<= p.Rc)
@inline evaluate_d(p::ShiftCutoff, r) = (@D p.pp(r)) .* (r .<= p.Rc)
# cutoff is automatically implemented because of field Rc





#########################################################
###         Lennard-Jones potential


"""`LennardJonesPotential <: PairPotential`

Implementation of the lennard-Jones potential    
   e0 * ( (r0/r)¹² - 2 (r0/r)⁶ )

Constructor: `LennardJonesPotential(;r0=1.0, e0=1.0)`
"""
type LennardJonesPotential <: PairPotential
    r0::Float64
    e0::Float64
end

LennardJonesPotential(; r0=1.0, e0=1.0) = LennardJonesPotential(r0, e0)


@inline evaluate(p::LennardJonesPotential, r) =
    p.e0 * ((p.r0./r).^12 - 2.0 * (p.r0./r).^6)
@inline evaluate_d(p::LennardJonesPotential, r) =
    -12.0 * p.e0 * ((p.r0./r).^12 - (p.r0./r).^6) ./ r


#########################################################
###         Simple Exponential
"""`SimpleExponential`

   A exp( B (r/r0 - 1) )
    
   TODO: this is acting as if it had 3 parameters, but there are actually only
      two - rewrite accordingly?
"""
type SimpleExponential <: PairPotential
    A::Float64
    B::Float64
    r0::Float64
end
@inline evaluate(p::SimpleExponential, r) =
    p.A * exp( p.B * (r/p.r0 - 1.0) )
@inline evaluate_d(p::SimpleExponential, r) =
    p.A*p.B/p.r0 * exp( p.B * (r/p.r0 - 1.0) )



#########################################################
###         Morse Potential
"""`MorsePotential <: PairPotential`

   e0 ( exp( -2 A (r/r0 - 1) ) - 2 exp( - A (r/r0 - 1) ) )
    
   TODO: this is acting as if it has 3 parameters, but there are actually only
      two -> rewrite accordingly?
"""
type MorsePotential <: PairPotential
    e0::Float64
    A::Float64
    r0::Float64
end
MorsePotential(A::Float64; e0=1.0, r0=1.0) = MorsePotential(e0, A, r0)
@inline morse_exp(p::MorsePotential, r) = exp(-p.A * (r/p.r0 - 1.0))
@inline function evaluate(p::MorsePotential, r) 
    e = morse_exp(p, r); return p.e0 * e .* (e - 2.0) end
@inline function  evaluate_d(p::MorsePotential, r)
    e = morse_exp(p, r);  return (-2.0 * p.e0 * p.A / p.r0) * e .* (e - 1.0) end
# @inline function  evaluate_both(p::MorsePotential, r) 
#     e = morse_exp(p, r)
#     return p.e0 * e .* (e - 2.0), (-2.0 * p.e0 * p.A) * e .* (e - 1.0)/p.r0 end


#########################################################
###         Gupta Potential


type EAMPotential <: SitePotential
    V::PairPotential
    rho::PairPotential
    embed::SimpleFunction
end

"embedding function for the Gupts potential"
type GuptaEmbed <: SimpleFunction
    xi
end
@inline evaluate(p::GuptaEmbed, r) = p.xi * sqrt(r)
@inline evaluate_d(p::GuptaEmbed, r) = 0.5*p.xi ./ sqrt(r)

"""`GuptaPotential`:
    E_i = A ∑_{j ≠ i} v(r_ij) - ξ ∑_i √ ρ_i
        v(r_ij) = exp[ -p (r_ij/r0 - 1) ]
        ρ_i = ∑_{j ≠ i} exp[ -2q (r_ij / r0 - 1) ]
"""
GuptaPotential(A, xi, p, q, r0, TC::Type, TCargs...)  =
    EAMPotential( TC( SimpleExponential(A, p, r0), TCargs... ),      # V
                  TC( SimpleExponential(1.0, 2*q, r0), TCargs...),   # rho
                  GuptaEmbed( xi ) )                                 # embed



end
