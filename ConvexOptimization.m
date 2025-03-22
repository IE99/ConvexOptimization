% Note: Sensitive to L, eta2, stability_H_modifier
clc; clear;

% Parameters
N = 1000; % Number of grid points
%dx = x(2) - x(1); % Grid spacing
% Regularization parameter for x
eta2 = 1e-3; % For sinusoid
%eta2 = 1e-2; % For box function
%eta2 = 1e-1; % For box function with L=0.5um - converges, but unintuitive sol'n
max_iter = 100; % Maximum number of iterations
tol_x = 1e-1; % Convergence tolerance on H field
tol_y = 1e-1; % Convergence tolerance on permittivity
tol_x = 1e-2; % Convergence tolerance on H field
tol_y = 1e-2; % Convergence tolerance on permittivity
%stability_H_modifier = 1e-5;
stability_H_modifier = 6e-5;

% Physical values
x = linspace(0, 1, N); 
lambda = 1550e-9;
L = 100e-9; % Length of the domain
c = 1;
e_min = 1;
e_max = 10;

% Natural units
lambda_n = lambda/L;
omega = 2*pi*c/lambda_n; % Angular frequency
xi = (omega / c)^2; % Eigenvalue parameter

wavelength_sinusoid = 1/10.5;
k_sinusoid = 2*pi*c/wavelength_sinusoid;
sigma = 1/8; % Controls how wide the envelope is
envelope = exp(- (x - 1/2).^2 / (2 * sigma^2));  % Gaussian envelope
sinusoid = sin(k_sinusoid * x);  % Sinusoidal signal
env_sinusoid = envelope .* sinusoid;

h1 = heaviside(x-1/3);
h2 = heaviside(x-2*1/3);
box_fn = h1-h2;

H_target = env_sinusoid*stability_H_modifier; %1e-6 is too much (bad sol'n) %5e-5 is infeasible %6e-5 is best
%H_target = box_fn*stability_H_modifier;
%plot(H_target)

% Initial guesses
y0 = ones(N, 1); % Initial guess for y (inverse permittivity)
x0 = H_target'; % Initial guess for x (magnetic field)

% s_prim = {ones(N,1), ones(1,1), ones(1,1)};
% s_dual = s_prim;
% J = {zeros(N,1,1), zeros(N,1,1), zeros(N,1,1)};
% eps = {ones(N,1,1), ones(N,1,1), ones(N,1,1)};
% mu = eps;
% %whos eps
% [A1, A2, m, e, b, Dx] = maxwell_matrices(omega, s_prim, s_dual, mu, eps, J);
% % A1 = A1(1:N, 1:N);
% % A2 = A2(1:N, 1:N);
% A = Dx;

% Create finite differences matrix
e = ones(N,1);
A = spdiags([-e e], [0 1], N, N);
A(N,1) = 1; % Periodic boundary conditions
A = sparse(A);

% Optimization loop
y = y0;
x_opt = x0;
for iter = 1:max_iter
    % Step 1: Optimize y (inverse permittivity) with bounds
    B = A * diag(A * x_opt);
    d = xi * x_opt;

    % Debugging: Check for NaN or Inf in B and d
    if any(isnan(B(:))) || any(isinf(B(:))) || any(isnan(d)) || any(isinf(d))
        error('Invalid values (NaN or Inf) detected in B or d. Check A and x_opt.');
    end

    cvx_begin
        variable y_new(N)
        minimize(norm(B * y_new - d))
        subject to
            1/e_max <= y_new <= 1/e_min; % Bounds on y (inverse permittivity)
    cvx_end

    % Debugging: Check for NaN or Inf in y_new
    if any(isnan(y_new)) || any(isinf(y_new))
        error('Invalid values (NaN or Inf) detected in y_new. Check optimization bounds and regularization.');
    end

    % Step 2: Optimize x (magnetic field)
    Y = diag(y_new);
    cvx_begin quiet
        variable x_new(N)
        minimize(norm(A * Y * A * x_new - xi * x_opt) + eta2 * norm(x_new - x_opt))
    cvx_end

    % Debugging: Check for NaN or Inf in x_new
    if any(isnan(x_new)) || any(isinf(x_new))
        error('Invalid values (NaN or Inf) detected in x_new. Check optimization bounds and regularization.');
    end

    % Check for convergence
    new_tolerance_y = norm(y_new - y);
    new_tolerance_x = norm(x_new/stability_H_modifier - x_opt/stability_H_modifier);
    disp(new_tolerance_y);
    disp(new_tolerance_x);
    disp("-------")
    if new_tolerance_y < tol_y && new_tolerance_x < tol_x
        break;
    end

    % Update variables
    y = y_new;
    x_opt = x_new;
end

% Compute permittivity (epsilon)
epsilon = 1 ./ y;
x_opt_au = x_opt / stability_H_modifier;

% Plot results
figure;
subplot(3, 1, 1);
plot(x, H_target/stability_H_modifier, 'r', 'LineWidth', 2);
title('Desired Magnetic Field (H_{target})');
xlabel('Position (a.u.)');
ylabel('H (a.u.)');
grid on;

subplot(3, 1, 2);
plot(x, epsilon, 'b', 'LineWidth', 2);
title('Optimized Permittivity (\epsilon_{r})');
xlabel('Position (a.u.)');
ylabel('\epsilon');
grid on;

subplot(3, 1, 3);
plot(x, x_opt_au, 'b', 'LineWidth', 2);
title('Resulting Magnetic Field H');
xlabel('Position (a.u.)');
ylabel('H (a.u.)');
grid on;

% Display convergence information
fprintf('Optimization converged in %d iterations.\n', iter);