#myplot()

PyPlot.matplotlib[:rc]("figure",figsize=(6,4))

using Interact
using Optim
using ForwardDiff

using tip3p
using grids


@pyimport IPython.display as ipydisp # enable showing web pages in IFrames, and direct HTML for imolecule
@pyimport quippy # quippy is the python wrapper for QUIP. it is somewhat ASE compatible

function miframe(url)
    ipydisp.IFrame(url, width=800, height=500)
end

# for imolecule 
@pyimport ase.io.cif as asecif
@pyimport ase.io.xyz as asexyz
@pyimport imolecule
@pyimport io as pio

# for preconditioned minimization
@pyimport ase.optimize.precon as ase_precon
@pyimport ase.constraints as ase_const

# shorthand for imolecule
function imolecule_draw(atoms)
    sio = pio.BytesIO()
    asexyz.write_xyz(sio, [atoms])
    ipydisp.HTML(imolecule.draw(sio[:getvalue](), format="xyz",  camera_type="orthographic", size=(300,200), display_html=false))
end

function imolecule_draw(atomlist::Array{Any})
    s = ""
    for atoms in atomlist
        sio = pio.BytesIO()
        asexyz.write_xyz(sio, [atoms])
        s = join(s,imolecule.draw(sio[:getvalue](), format="xyz",  camera_type="orthographic", size=(300,200), display_html=false))
    end
    ipydisp.HTML(s)
end

# for drawing patches in matplotlib
@pyimport matplotlib.patches as patch
function patch_h2o(p = [0.0,0.0])
    ax = gca()
    ax[:set_aspect]("equal")
    ax[:add_artist](patch.Rectangle([0,-0.05]+p, 0.9,0.1, fc="grey", fill=true))
    ax[:add_artist](patch.Rectangle([0.05,-0.05]+p, 0.9,0.1, 104, fc="grey", fill=true))
    ax[:add_artist](patch.Circle([0,0]+p, 0.25, fc="red", fill=true))
    ax[:add_artist](patch.Circle([0.96,0.0]+p, 0.15, fc="white", fill=true))
    ax[:add_artist](patch.Circle([-0.24,0.93]+p, 0.15, fc="white", fill=true))
end

# water monomer geometry
θ0 = 104.4/180*π
r0 = 0.96
h2o_p0 = [0 0 0 ; r0 0 0 ; r0*cos(θ0) r0*sin(θ0) 0]

function make_h2o(p=[0.0 0.0 0.0])
    h2o = quippy.Atoms(n=3, lattice=[20 0 0 ; 0 20 0 ; 0 0 20], numbers=[8,1,1], positions=h2o_p0+repeat(p,outer=[3,1]));
end

function make_h4o2(p = [0.0 0.0 0.0]; optim=false)
    h4o2 = make_h2o(p)
    h2o = make_h2o(p+[2.5 0 0]) 
    h4o2[:add_atoms](h2o)

    if optim
        h4o2[:rattle](0.1)
        res = optimize(tip3p.potential, (p,s)->s[:] = tip3p.gradient(p), vec(h4o2[:get_positions]()'), ftol=1e-4, grtol=1e-2)
        h4o2[:set_positions](reshape(res.minimum, (3,6))')
    end
    h4o2
end

function make_h12o6()
    quippy.Atoms(n=18, lattice=[20 0 0 ; 0 20 0 ; 0 0 20], numbers=repeat([8,1,1], outer=[6]),
                        positions=[ h2o_p0 ; 
                                    h2o_p0 +  repeat([3.5 0.0 0.0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([5.0 2.5 0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([3.5 5.0 0.0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([0.0 5.0 0.0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([-1.5 2.5 0], outer=[3,1])]);
end

# radius of gyration of water hexamer
function h12o6_Rg(x)
    rO = reshape(x, (3,18))[:,1:3:18]; mr = mean(rO, 2)
    sqrt(mean(sum((rO-repeat(mr, inner=[1,6])).^2,1)))
end

# velocity Verlet LAngevin integrator

const kB = 8.6173324e-5 # Boltzmann constant in eV/K

function velocityVerletLangevin!(q::Vector{Float64}, p::Vector{Float64}, Fq::Vector{Float64}, F, m::Vector{Float64}, h::Float64, T::Float64, gamma::Float64, ph2::Vector{Float64}, qh2::Vector{Float64}, r::Vector{Float64})
    N = length(q)
    r[:] = randn(N)
    expmgammah = exp(-gamma*h)
    sqrtkBT1mexpm2gammah = sqrt(kB*T*(1-exp(-2.0*gamma*h)))
    @inbounds @simd for i=1:N
        ph2[i] = p[i] + (0.5*h)*Fq[i]
        qh2[i] = q[i] + (0.5*h)*ph2[i]/m[i]
        ph2[i] = expmgammah*ph2[i] + sqrtkBT1mexpm2gammah*r[i]*sqrt(m[i])
        q[i]  = qh2[i] + (0.5*h)*ph2[i]/m[i]
    end
    Fq[:] = F(q)
    @inbounds @simd for i=1:N
        p[i]  = ph2[i] + (0.5*h)*Fq[i]
    end
    return
end


# water dimer dissociation curve
function water_dimer_dissoc()
    r = 2.3:0.005:8.0
    h4o2 = make_h4o2(optim=true)
    p0 = h4o2[:get_positions]()
    p = copy(p0)
    dOO = p0[4,:]- p0[1,:] 
    Edim_tip3p = zeros(r)
    for i=1:length(r)
        p[4:6,:] = p0[4:6,:] + repeat((r[i]/norm(dOO)-1.0)*dOO, outer=[3,1])
        Edim_tip3p[i] = tip3p.potential(vec(p'))
    end
    r, Edim_tip3p
end



function water_hexamer_dynamics(;T::Float64, Nsteps=1000 Nsubsamp=1 )
    h12o6 = make_h12o6()

    h=0.5
    m = repeat(h12o6[:get_masses](), inner=[3])*quippy.units[:MASSCONVERT] #masses to conform to eV, A, fs units
    q = vec(h12o6[:get_positions]()')
    Fq = -tip3p.gradient(q)
    p = zeros(q) ; tmp1 = zeros(q); tmp2 = zeros(q); tmp3 = zeros(q)
 
    Rg = zeros(Nsteps÷Nsubsamp); w1z = zeros(Nsteps÷Nsubsamp); n=1

    println("starting $Nsteps iterations")
    for i=1:Nsteps
        if i%Nsubsamp == 1
            Rg[n] = h12o6_Rg(q)
            w1z[n] = q[6]-q[3]
            n = n+1
        end
        if i%10000 == 1
            println("iteration $i")
        end
        velocityVerletLangevin!(q, p, Fq, q->(-tip3p.gradient(q)), m, h, T, 0.5, tmp1, tmp2, tmp3)   
    end
    println("finished $Nsteps iterations")
    Rg,w1z
end

function mean_err_corr(x; acorr_limit=0)
    r = (acorr_limit == 0 ? (1:length(x)÷2) : (1:acorr_limit) )
         
    C = fftshift(xcorr(x - mean(x), x - mean(x)))[r]
    mean(x), std(x)/sqrt(length(x)/(1+sum(C/C[1])))
end
;