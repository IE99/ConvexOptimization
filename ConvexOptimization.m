%% Parameter Definitions
clc; clear;

% Physical values
N = 1000; % Number of grid points
x = linspace(0, 1, N); % Position
lambda = 1550e-9; % Resonant frequency
L = 100e-9; % Length of the domain
c = 1; % Speed of light
e_min = 1; % Minimum permittivity
e_max = 10; % Maximum permittivity
max_iter = 100; % Maximum number of iterations
tol_x = 1e-1; % Convergence tolerance on H field
tol_y = 1e-1; % Convergence tolerance on inverse permittivity

H_function = "box"; % "sinusoid", "box", or "sawtooth"

% Generate the target magnetic field functions
%   A stability_H_modifier is used for stability of the optimization
%   to have a well-conditioned matrix. This works because H is in arbitrary
%   units. The actual H-field amplitude depends on the source field
%
%   eta, the regularization parameter, is set by trail and error for each
%   function
if H_function == "sinusoid"
    % Function-specific parameters
    eta = 1e-3;
    stability_H_modifier = 6e-5;

    % Target field
    wavelength_sinusoid = 1/10.5;
    k_sinusoid = 2*pi*c/wavelength_sinusoid;
    sigma = 1/8; % Controls how wide the envelope is
    envelope = exp(- (x - 1/2).^2 / (2 * sigma^2));  % Gaussian envelope
    sinusoid = sin(k_sinusoid * x);  % Sinusoidal signal
    env_sinusoid = envelope .* sinusoid;

    H_target = env_sinusoid*stability_H_modifier;

elseif H_function == "box"
    % Function-specific parameters
    eta = 1e-2; % Regularization parameter for x
    stability_H_modifier = 1e-5;

    % Target field
    h1 = heaviside(x-0.25);
    h2 = heaviside(x-0.75);
    box_fn = h1-h2;
    
    H_target = box_fn*stability_H_modifier;

elseif H_function == "sawtooth"
    % Function-specific parameters
    eta = 1e-2; % Regularization parameter for x
    stability_H_modifier = 1e-5;

    % Target field
    sawtooth_fn = zeros(size(x));
    sawtooth_fn(x > 0.25 & x <= 0.75) = (x(x > 0.25 & x <= 0.75) - 0.25) / 0.5; % Linear ramp from 0 to 1

    H_target = sawtooth_fn*stability_H_modifier;
end

% Parameters in natural units
lambda_n = lambda/L;
omega = 2*pi*c/lambda_n; % Angular frequency
xi = (omega / c)^2; % Eigenvalue parameter

% Initial guesses
%y0 = ones(N, 1); % Initial guess for y (inverse permittivity)
y0 = 1/e_max + (1/e_min - 1/e_max) * rand(1, N); % Random guess for initial y
x0 = H_target'; % Initial guess for x (magnetic field)

% Create finite differences matrix
e = ones(N,1);
A = spdiags([-e e], [0 1], N, N);
A(N,1) = 1; % Periodic boundary conditions
A = sparse(A);

%% Optimization loop
y = y0;
x_opt = x0;
for iter = 1:max_iter
    % Optimize y (inverse permittivity) with bounds
    B = A * diag(A * x_opt);
    d = xi * x_opt;

    cvx_begin
        variable y_new(N)
        minimize(norm(B * y_new - d))
        subject to
            1/e_max <= y_new <= 1/e_min; % Bounds on y (inverse permittivity)
    cvx_end

    % Optimize x (magnetic field)
    Y = diag(y_new);
    cvx_begin quiet
        variable x_new(N)
        minimize(norm(A * Y * A * x_new - xi * x_opt) + eta * norm(x_new - x_opt))
    cvx_end

    % Check for convergence
    new_tolerance_y = norm(y_new - y);
    new_tolerance_x = norm(x_new/stability_H_modifier - x_opt/stability_H_modifier);
    if new_tolerance_y < tol_y && new_tolerance_x < tol_x
        break;
    end

    % Update variables
    y = y_new;
    x_opt = x_new;
end

%% Calculate Results
% Calculate permittivity
epsilon = 1 ./ y;

% Modify arbitrary units for plot clarity
H_opt_au = x_opt / stability_H_modifier;
x = linspace(-10, 10, N); % To easily compare against PSO

% Get filtered responses (structure that could be fabricated)
epsilon_filtered = Filter(epsilon, N);

% Find resulting field from filtered structure using same optimization we
% used before - should give same result as if we did an FDTD simulation,
% but will be faster
y_filtered = 1 ./ epsilon_filtered;
Y = diag(y_filtered);
for iter = 1:max_iter
    cvx_begin
        variable x_filtered(N)
        minimize(norm(A * Y * A * x_filtered - xi * x_opt) + eta * norm(x_filtered - x_opt))
    cvx_end

    tol_x = norm(x_filtered/stability_H_modifier - x_filtered/stability_H_modifier);
    if tol_x < 1e-2 % Break when we get effectively converge
        break;
    end

    x_opt = x_filtered;
end
x_filtered = x_filtered / stability_H_modifier;

%% Plot results
figure;
subplot(2, 1, 1);
plot(x, H_target/stability_H_modifier, 'r', 'LineWidth', 2); hold on;
plot(x, H_opt_au, 'b', 'LineWidth', 2);
plot(x, x_filtered, 'g--', 'LineWidth', 2);
legend('Target H', 'Resulting H', 'Resulting H_{filtered}');
xlabel('Position (a.u.)');
ylabel('H (a.u.)');
title('Magnetic Field (H)');
grid on;
hold off;

subplot(2, 1, 2);
plot(x, epsilon, 'b', 'LineWidth', 2); hold on;
plot(x, epsilon_filtered, 'g--', 'LineWidth', 2);
legend('Optimized \epsilon_{r}', 'Filtered \epsilon_{r}');
title('Permittivity (\epsilon_{r})');
xlabel('Position (a.u.)');
ylabel('\epsilon_{r}');
grid on;
hold off;

% Display convergence information
fprintf('Final tolerance in y: %d.\n', new_tolerance_y);
fprintf('Final tolerance in x: %d.\n', new_tolerance_x);
fprintf('Optimization converged in %d iterations.\n', iter);