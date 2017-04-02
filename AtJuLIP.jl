module AtJuLIP

import JuLIP
import AtomsInterface
import ASE

type JuLIPCalculator
   calc::AtomsInterface.AbstractCalculator
end


j2a(at::JuLIP.AbstractAtoms) = ASE.ASEAtoms(at.po)


JuLIP.energy(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms)
   = AtomsInterface.potential_energy(j2a(at), calc.calc)

JuLIP.forces(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms)
   = AtomsInterface.forces(j2a(at), calc.calc)

JuLIP.Potentials.site_energy(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms, i0::Int)
   = AtomsInterface.site_energy(i0, j2a(at), calc.calc)

JuLIP.Potentials.site_energy_d(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms, i0::Int)
   = scale!(AtomsInterface.site_forces(i0, j2a(at), calc.calc), -1)

end
