%% Uniform random sample in the range [a..b]
function res = uniform (a, b, n)

         res = a + rand(n,1)*(b-a);
