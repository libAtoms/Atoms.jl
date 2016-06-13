

push!(LOAD_PATH, ".", "..")

using TestAtoms, Potentials, MatSciPy, ASE, MDTools


###### TIMING TESTS
println("Timing for PairCalculator")
at = repeat(bulk("Al"; cubic=true), (200, 200, 1))
set_pbc!(at, (false, false, false))
p = ShiftCutoff(LennardJonesPotential(r0=2.8), 6.0)
calc = MatSciPy.PairCalculator(p)
println("Cost of one neighbourlist")
@time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
@time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
@time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
@time i,r,R = MatSciPy.neighbour_list(at, "idD", cutoff(p))
# println("cost of one force assembly (incl neighbour_list cost")
# @time f = potential_energy_d(at, calc)
# @time f = potential_energy_d(at, calc)
# @time f = potential_energy_d(at, calc)

println("Julia-internal neighbourlist")
X = positions(at)[1:2, 1:end]
@time nlist = MDTools.NeighbourList(X, cutoff(p))
@time nlist = MDTools.NeighbourList(X, cutoff(p))
@time nlist = MDTools.NeighbourList(X, cutoff(p))
@time nlist = MDTools.NeighbourList(X, cutoff(p))

