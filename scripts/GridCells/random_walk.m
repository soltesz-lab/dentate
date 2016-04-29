%# Set the parameters
LB_X = 1; %# Lower bound
UB_X = 30; %# Upper bound
LB_Y = 1; %# Lower bound
UB_Y = 30; %# Upper bound

T  = 100; %# Number of observations
N  = 3; %# Number of samples
X0 = 1.5 * LB_X; %# Arbitrary start point near LB
Y0 = 1.5 * LB_Y; %# Arbitrary start point near LB

Gain_X = 1.5;
Gain_Y = 1.0;

%# Generate the jumps
Jump_X = randn(N, T-1) * Gain_X;
Jump_Y = randn(N, T-1) * Gain_Y;

%# Build the constrained random walk
X = X0 * ones(N, T);
for t = 2:T
    X(:, t) = max(min(X(:, t-1) + Jump_X(:, t-1), UB_X), 0);
end
X = X';
Y = Y0 * ones(N, T);
for t = 2:T
    Y(:, t) = max(min(Y(:, t-1) + Jump_Y(:, t-1), UB_Y), 0);
end
Y = Y';
