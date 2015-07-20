module gp

export gpr, CovG, CovL, CovGsin, CovLsin, CovDot, CovDotWeight

CovG = ( (x1, x2, sig::Float64) -> exp(- sumabs2(x1-x2)/(2.0*sig^2)))
CovGsin = ( (x1, x2, sig::Float64) -> exp(- 2.0*sumabs2(sin((x1-x2)/2.0))/sig^2))
CovL = ( (x1, x2, sig::Float64) -> exp(- sumabs(x1-x2)/sig))
CovLsin = ( (x1, x2, sig::Float64) -> exp(- sumabs(sin((x1-x2)/2.0))/sig))
CovDot = ( (x1, x2, sig::Float64) -> (x1 ⋅ x2)^sig)
CovDotWeight = ( (x1, x2, sig::Float64) -> (sum(x1.*x2.*(2.0.^(-[1:length(x1)]/sig))))^4.0 )





#
# 1D special case
#
function gpr(x::Array{Float64,1}, y::Array{Float64,1}, xp::Array{Float64,1}, Cov, noise::Float64, len::Float64)
    gpr(reshape(x, 1, length(x)), y, reshape(xp, 1, length(xp)), Cov, noise, len)
end

#
# data is a matrix : (Ndescriptors,Ndata)
#
function gpr(x::Array{Float64,2}, y::Array{Float64,1}, xp::Array{Float64,2}, Cov, noise::Float64, len::Float64)
    Ndata = size(x,2);  # number of training data points
    K=zeros(Ndata,Ndata);       # kernel
    # build kernel matrix
    for i=1:Ndata
        for j=1:Ndata
            K[i,j] = Cov(x[:,i], x[:,j], len)/(sqrt(Cov(x[:,i],x[:,i],len))*sqrt(Cov(x[:,j], x[:,j], len)))+noise^2*(i==j)
        end
    end
    a = inv(K)*y # invert kernel matrix and compute coefficients

    Np = size(xp,2) # number of prediction points
    yp = zeros(Np)  # predicted values
    k = zeros(Ndata)    # covariance of new points
    for i=1:Np
        for j=1:Ndata
            k[j] = Cov(xp[:,i], x[:,j], len)/(sqrt(Cov(xp[:,i],xp[:,i],len))*sqrt(Cov(x[:,j], x[:,j], len)))
        end
        yp[i] = k⋅a
    end
    return yp
end



end
