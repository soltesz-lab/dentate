function r=DG_spike_gen(s,SigmaZ,I)
%s should be N time bins x P cells
[N,P]=size(s)

s0=s;
s=zeros(N,I,P);
for i=1:I,
    s(:,i,:)=s0;
end

z=zeros(N,I,P);
for i=1:I
    z(:,i,:)=mvnrnd(zeros(N,P),SigmaZ);
end

y = (s+z) > 0;
[i,j,s] = find(y);
[m,n] = size(y);
r = sparse(i,j,s,m,n);

