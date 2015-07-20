module pbc

global L = 0.0 # size of periodic cubic unit cell, 0 means no periodicity

# compute the box size given the number of particles and the density 
function cubesize(N, rho)
    return (N/rho)^(1.0/3.0)
end

function set_density(N, rho)
    global L
    L = cubesize(N, rho)
end


# remap elements of a difference vector into -0.5 L < dr < 0.5 L
function min_image!(dr::Array{Float64,1})
    @assert L!=0.0 "L==0.0 in pbc.min_image!"
    for i=1:length(dr)
        #dr[i] += ((dr[i]<-0.5*L)-(dr[i]>0.5*L))*L
        if dr[i] < -0.5*L
            dr[i] += L
        elseif dr[i] > 0.5*L
            dr[i] -= L
        end
        
        #dr[i] -= L*floor(dr[i]/L+0.5)
    end
    #return dr
    #dr += ((dr.<-0.5*L)-(dr.>0.5*L))*L
    #dr -= L*floor(dr/L+0.5)
    #return dr
end

function map_into_cell!(x::Array{Float64,2})
    @assert L!=0.0 "L==0.0 in pbc.min_image!"
    for i=1:size(x,2)
        for j=1:size(x,1)
            #x[j,i] += ((x[j,i]<-0.5*L)-(x[j,i]>0.5*L))*L;
            if x[j,i] < -0.5*L
                x[j,i] += L;
            elseif x[j,i] > 0.5*L
                x[j,i] -= L;
            end
        end
    end
    #return x;
end




end
