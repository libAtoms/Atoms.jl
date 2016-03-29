module tip3p

using ForwardDiff

function h2o_quad(p)
    bond = [24.1648  -50.3605    26.2387   ]     # coefficients for bond A angstroms
    angle = [7.207    -0.137997   0.000660598]   # coefficient for angle in degrees
    
    r12 = sqrt(sumabs2(p[4:6]-p[1:3]))
    r13 = sqrt(sumabs2(p[7:9]-p[1:3]))
    θ = acos(sum((p[4:6]-p[1:3]).*(p[7:9]-p[1:3]))/(r12*r13))/π*180.0
    E = bond[1]+bond[2]*r12+bond[3]*r12*r12 + bond[1]+bond[2]*r13+bond[3]*r13*r13 + angle[1]+angle[2]*θ+angle[3]*θ*θ
end


function potential{T}(p::Vector{T})

    # input vector of positions in this orderL Ox Oy Oz Hx Hy Hz Hx Hy Hz Ox Oy Oz...
    
    qO = -0.8340
    qH = 0.4170
    σ = 3.15061       # in Angstrom
    ε = 0.6364 * 0.01 # conversion from kJ/mol to eV
    bohr = 0.529177249 # 1 Bohr in Angstrom
    hartree = 27.2114  # in eV
    
    E = 0.0
    
    for i=1:length(p)÷9                          # loop over oxygens (i.e. H2O monomers)
        E += h2o_quad(p[(i-1)*9+1:(i-1)*9+9])    # monomer energy
        for j=i+1:length(p)÷9                    # loop over other oxygens
            is = (i-1)*9; js = (j-1)*9
            O1x = p[is+1]; O1y = p[is+2]; O1z = p[is+3]
            H1O1x = p[is+4]; H1O1y = p[is+5]; H1O1z = p[is+6]
            H2O1x = p[is+7]; H2O1y = p[is+8]; H2O1z = p[is+9]
            O2x = p[js+1]; O2y = p[js+2]; O2z = p[js+3]
            H1O2x = p[js+4]; H1O2y = p[js+5]; H1O2z = p[js+6]
            H2O2x = p[js+7]; H2O2y = p[js+8]; H2O2z = p[js+9]
            rij = sqrt(sumabs2(p[is+1:is+3]-p[js+1:js+3]))
            E += 4.0*ε*((σ/rij)^12-(σ/rij)^6)     # LJ term
            # Electrostatics
            Es = 0.0
            Es += qO*qO/rij
            Es += qO*qH*(1/sqrt(sum((O1x-H1O2x)^2+(O1y-H1O2y)^2+(O1z-H1O2z)^2))
                        +1/sqrt(sum((O1x-H2O2x)^2+(O1y-H2O2y)^2+(O1z-H2O2z)^2))
                        +1/sqrt(sum((O2x-H1O1x)^2+(O2y-H1O1y)^2+(O2z-H1O1z)^2))
                        +1/sqrt(sum((O2x-H2O1x)^2+(O2y-H2O1y)^2+(O2z-H2O1z)^2)))
            Es += qH*qH*(1/sqrt(sum((H1O1x-H1O2x)^2+(H1O1y-H1O2y)^2+(H1O1z-H1O2z)^2))
                        +1/sqrt(sum((H1O1x-H2O2x)^2+(H1O1y-H2O2y)^2+(H1O1z-H2O2z)^2))
                        +1/sqrt(sum((H2O1x-H1O2x)^2+(H2O1y-H1O2y)^2+(H2O1z-H1O2z)^2))
                        +1/sqrt(sum((H2O1x-H2O2x)^2+(H2O1y-H2O2y)^2+(H2O1z-H2O2z)^2)))
            E += Es*hartree*bohr
        end
    end
    E
end

h2o_quad_gradient = ForwardDiff.gradient(h2o_quad)
gradient = ForwardDiff.gradient(potential)

end