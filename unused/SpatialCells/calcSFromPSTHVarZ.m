function s=calcSFromPSTHVarZ(rbar,varz);
%Assumes theta = 0

o=optimset('TolX',0.01);

N=length(rbar);
for n=1:N,
    if rbar(n),
        s(n) = fminbnd(@(x) calcRBarFromSVarZ(x,varz,rbar(n)),-5,5,o);
    else
        s(n)=-Inf;
    end
end
s=s(:);

function err=calcRBarFromSVarZ(s,varz,rbar0);
rbar=normcdf(s, 0, sqrt(varz));
err=abs(rbar-rbar0);
