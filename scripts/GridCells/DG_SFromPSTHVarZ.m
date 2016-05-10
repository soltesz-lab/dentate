function s=DG_SFromPSTHVarZ(rbar,varz);
%Assumes theta = 0

[N,P]=size(rbar)
o=optimset('TolX',0.1);
s=zeros(N,P);

for n=1:N
    n
    tic
    for p=1:P
        sp = myfun(rbar,varz,o,n,p);
        s(n,p) = sp;
    end
    toc
end

function s=myfun(rbar,varz,o,n,p)
    if rbar(n,p)
        s = fminbnd(@(x) calcRBarFromSVarZ(x,varz,rbar(n,p)),-5,5,o);
    else
        s=-Inf;
    end

function err=calcRBarFromSVarZ(s,varz,rbar0);
rbar=normcdf(s, 0, sqrt(varz));
err=abs(rbar-rbar0);
