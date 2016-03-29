module lj

import pbc;

global conf = 0.0;    # strength of confining potential 
global cutoff = 0.0;  # cutoff of the LJ potential for the periodic case, 0 means no cutoff
global cutoff2 = cutoff^2;

global Vljshift = 0.0;
# set Vljshift = Vlj(cutoff) to have no jump condition at the cutoff

function info()
    return "Lennard-Jones interaction\n  V(r)=(1/r)^12-(1/r)^6\n  parameters:\n  cutoff = $cutoff\n  conf = $conf\n  shift = $Vljshift"*(pbc.L==0.0?"\n  using no PBC":"\n  using PBC with L = $(pbc.L)")
end

function set_shiftedLJ()
    global Vljshift
    Vljshift = 0.0
    if cutoff > 0.0
        Vljshift = Vlj(cutoff)
    end
end

function set_confinement(c)
    global conf
    conf = c
end

function set_cutoff(c)
    global cutoff
    global cutoff2
    cutoff = c
    cutoff2 = cutoff^2
end

#
# energy of LJ system 
# x is (N,dim) array of positions
#
function energy(x)
    N = size(x,2);
    e = 0.0;
    for i=1:N
        for j=i+1:N
            dr = x[:,i]-x[:,j];
            if pbc.L > 0.0
                pbc.min_image!(dr);
            end
            e += Vlj2(sumabs2(dr));
        end
    end
    if conf != 0.0
        e += conf*sumabs2(x);
    end
    return e+(rand()-0.5)*1e-12;
end

#
#
# force on particles
# x is (N,dim) array of positions
#
function force(x)
    N = size(x,2)
   
    f = zeros(x)
    for i = 1:N
        for j=i+1:N
            dr = x[:,j]-x[:,i]
            r = norm(dr)
            df = Vljder(r)*(dr/r)
            f[:,i] += df
            f[:,j] -= df
        end
    end
    
    return f
end

#
# energy difference of LJ system upon moving particle i by vector dx
#
function energy1(x, i, dx)
    N = size(x, 2)
    denergy = 0.0;
    xi = x[:,i];
    for j=1:N
        if i==j
            continue
        end
        dr1 = x[:,j]-xi;
        dr2 = dr1-dx;
        if pbc.L > 0.0
            pbc.min_image!(dr1);
            pbc.min_image!(dr2);
        end
        denergy += Vlj2(sumabs2(dr2)) - Vlj2(sumabs2(dr1));
    end
    if conf != 0.0
        denergy += conf*(sumabs2(x[:,i]+dx)-sumabs2(x[:,i]));
    end
    return denergy+(rand()-0.5)*1e-8;
end

#
# LJ pair potential
#
function Vlj(r)
    if cutoff > 0.0 && r > cutoff
        return 0.0;
    end
    return (1/r)^12-(1/r)^6-Vljshift;
end

#
# derivative of LJ potential
#

function Vljder(r)
    if cutoff >0.0 && r > cutoff
        return 0.0
    end
    return -12.0*(1/r)^13+6.0*(1/r)^7
end

      
        
#
# LJ pair potential operating on r^2
#
function Vlj2(r2)
    oner6 = (1/r2)^3;
    return (cutoff2 == 0.0 || r2 < cutoff2) * (oner6*oner6-oner6-Vljshift);
end

end
