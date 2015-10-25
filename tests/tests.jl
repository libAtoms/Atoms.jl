

push!(LOAD_PATH, ".", "..")

using TestAtoms, Potentials, MatSciPy, ASE


# println("Testing Lennard-Jones Potential")
# test_ScalarFunction(LennardJonesPotential(), 0.9 + rand(20))

# println("Testing Lennard-Jones Potential with cutoff")
# test_ScalarFunction(SWCutoff(LennardJonesPotential(), 2.1, 1.0), 1.5 + rand(20))


# println("Testing the Components of the Gupta Potential")
# println("1. the SimpleExponential: ")
# test_ScalarFunction(SimpleExponential(1.234, 2.345, 1.321), 0.5 + rand(20))
# println("2. the EmbeddingFunction: ")
# test_ScalarFunction(GuptaEmbed(1.234), 0.5 + rand(20))


# println("Testing PairCalculator with LennardJonesPotential")
# p = SWCutoff(LennardJonesPotential(r0=2.8), 6.0, 1.0)
# at = repeat(bulk("Al"; cubic=true), (2,2,2))
# X = positions(at)
# set_positions!(at, X + 0.2 * rand(size(X)))
# test_potentialenergy(MatSciPy.PairCalculator(p), at)



# ## TODO: double-check this one is working?!
# println("Testing EAMCalculator with GuptaPotential")
# p = GuptaPotential(1.234, 0.987/12, 2.345, 1.432, 2.789,
#                    SWCutoff, 6.0, 1.0)
# # test_ScalarFunction(p.rho, 2.6 + 2.0 * rand(20))
# # test_ScalarFunction(p.embed, 1.0 + 2.0 * rand(20))
# at = repeat(bulk("Al"; cubic=true), (2,2,2))
# X = positions(at)
# set_positions!(at, X + 0.2 * rand(size(X)))
# test_potentialenergy(MatSciPy.EAMCalculator(p), at)




# reference configuration
at = bulk("Al")
at = repeat(at, (5, 5, 1))
X0 = positions(at);
# jiggle the positions
set_positions!(at, X0+0.03 * rand(size(X0)))
# create a calculator
r0 = rnn("Al")
rcut = 2.5 * r0
calc = MatSciPy.PairCalculator( SWCutoff( LennardJonesPotential(r0, 1.0), rcut, 1.0 ) )
test_potentialenergy(calc, at)
dE = potential_energy_d(at, calc)
frc = forces(at, calc)
println(vecnorm(dE + frc))

# test_ScalarFunction(SWCutoff( LennardJonesPotential(r0, 1.0), rcut, 1.0 ), 2.4 + rand(20))





###### TIMING TESTS
# println("Timing for PairCalculator")
# at = repeat(bulk("Al"; cubic=true), (20,20,20))
# p = ShiftCutoff(LennardJonesPotential(r0=2.8), 6.0)
# calc = MatSciPy.PairCalculator(p)
# println("Cost of one neighbourlist")
# @time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
# @time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
# @time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
# println("cost of one force assembly (incl neighbour_list cost")
# @time f = potential_energy_d(at, calc)
# @time f = potential_energy_d(at, calc)
# @time f = potential_energy_d(at, calc)



