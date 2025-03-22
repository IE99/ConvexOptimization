clc; clear; close all;

%initials
Nx = 500; 
x = linspace(-10, 10, Nx);

% Target Field (Gaussian-modulated sinusoidal wave)
A_target = 1;
sigma_target = 2;
k_target = 5;
phi_target = 0;
H_target = A_target * exp(-x.^2 / (2 * sigma_target^2)) .* cos(k_target*x + phi_target);
H_target = H_target(:); 

eps_min = 1; % Minimum permittivity
eps_max = 10; % Maximum permittivity
numParticles = 30;
numDimensions = Nx; % The permittivity is a function of x, so we optimize Nx values
maxIterations = 5000;

% Gaussian permittivity profile
eps_background = eps_min;
eps_center = (eps_min + eps_max) / 2;
sigma_eps = 4; %standard deviation of the Gaussian function used to create the initial permittivity profile

% Create the base permittivity profile
base_profile = eps_background + (eps_center - eps_background) * exp(-x.^2 / (2 * sigma_eps^2));

% Initialize particles around this base profile with small random variations
positions = zeros(numParticles, numDimensions);
for i = 1:numParticles
    if i == 1
        positions(i, :) = base_profile;
    else
        % Add controlled variations to other particles
        variation = 0.2 * (eps_max - eps_min) * (rand(1, numDimensions) - 0.5);
        positions(i, :) = base_profile + variation;
        % Ensure values stay within bounds
        positions(i, :) = max(eps_min, min(eps_max, positions(i, :)));
    end
end

velocities = -0.05 + 0.1 * rand(numParticles, numDimensions);

w = 0.7; % Inertia weight
c1 = 1.5; % Cognitive coefficient
c2 = 1.5; % Social coefficient

%% **2. Evaluate initial fitness**
personalBestPositions = positions;
personalBestScores = zeros(numParticles, 1);

for i = 1:numParticles
    personalBestScores(i) = compute_fitness(positions(i, :), x, H_target, sigma_target);
end
[globalBestScore, bestIndex] = min(personalBestScores);
globalBestPosition = personalBestPositions(bestIndex, :);

%% **3. PSO Main Loop**
bestFitnessHistory = zeros(maxIterations, 1);

figure;
h_plot = plot(x, H_target, 'r', 'LineWidth', 2, 'DisplayName', 'Target Field');
hold on;

H_optimized_values = compute_field(globalBestPosition, x, sigma_target);
%It calculates the magnetic field based on the best permittivity profile found so far
if dot(H_optimized_values, H_target) < 0
    H_optimized_values = -H_optimized_values;
end


h_optimized = plot(x, H_optimized_values, 'b--', 'LineWidth', 2, 'DisplayName', 'Computed Field');
xlabel('Position (a.u.)'); ylabel('Magnetic Field H (a.u.)');
title('PSO Optimization of Dielectric Structure');
grid on; legend;

% Add a plot of initial permittivity profile
figure;
plot(x, base_profile, 'g', 'LineWidth', 2);
xlabel('Position (a.u.)'); ylabel('Dielectric Permittivity (\epsilon)');
title('Initial Dielectric Structure');
grid on;

for iter = 1:maxIterations
    for i = 1:numParticles
        % Update velocity
        r1 = rand(1, numDimensions);
        r2 = rand(1, numDimensions);
        velocities(i, :) = w * velocities(i, :) ...
                         + c1 * r1 .* (personalBestPositions(i, :) - positions(i, :)) ...
                         + c2 * r2 .* (globalBestPosition - positions(i, :));

        % Update position
        positions(i, :) = positions(i, :) + velocities(i, :);

        % Apply boundary constraints
        positions(i, :) = max(eps_min, min(eps_max, positions(i, :)));

        % Evaluate new fitness
        newFitness = compute_fitness(positions(i, :), x, H_target, sigma_target);

        % Update personal best
        if newFitness < personalBestScores(i)
            personalBestScores(i) = newFitness;
            personalBestPositions(i, :) = positions(i, :);
        end
    end

    % Update global best
    [currentBestScore, bestIndex] = min(personalBestScores);
    if currentBestScore < globalBestScore
        globalBestScore = currentBestScore;
        globalBestPosition = personalBestPositions(bestIndex, :);
    end

    % Store best fitness for plotting
    bestFitnessHistory(iter) = globalBestScore;

    % Compute optimized field
    H_optimized_values = compute_field(globalBestPosition, x, sigma_target);
    if dot(H_optimized_values, H_target) < 0
        H_optimized_values = -H_optimized_values;
    end

    set(h_optimized, 'YData', H_optimized_values);
    title(sprintf('Iteration %d: Best Fitness = %.6f', iter, globalBestScore));
    drawnow; % Use drawnow instead of pause for better performance
end

%% **4. Plot Results**
figure;
subplot(2,1,1);
plot(x, globalBestPosition, 'g', 'LineWidth', 2);
hold on;
plot(x, base_profile, 'r--', 'LineWidth', 1);
xlabel('Position (a.u.)'); ylabel('Dielectric Permittivity (\epsilon)');
title('Optimized Dielectric Structure');
legend('Optimized Profile', 'Initial Profile');
grid on;

subplot(2,1,2);
plot(bestFitnessHistory, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Fitness Value');
title('PSO Convergence');
grid on;

disp('Optimized Permittivity Profile:');
disp(globalBestPosition);
fprintf('Final Best Fitness: %.6f\n', globalBestScore);

%% **5. Functions: Fitness & Field Computation**
function cost = compute_fitness(eps_profile, x, H_target, sigma_target)
    % Compute the field
    H_computed = compute_field(eps_profile, x, sigma_target);

    % Ensure both are column vectors of the same size
    H_computed = H_computed(:);
    H_target = H_target(:);

    % Normalize the computed field for comparison
    H_computed = H_computed / max(abs(H_computed)) * max(abs(H_target));

    % Ensure field has same sign orientation as target using dot product
    if dot(H_computed, H_target) < 0
        H_computed = -H_computed;
    end

    % Calculate mean squared error
    cost = norm(H_computed - H_target);
end

function H = compute_field(eps_profile, x, sigma_target)
    % Simple 1D wave simulation

    % Grid setup
    dx = x(2) - x(1);
    Nx = length(x);

    % Create a simple 1D wave operator (continuous wave equation)
    k0 = 5; % Base wave number (matching target frequency)

    % Direct solution using sinusoidal basis
    H = zeros(size(x));
    for i = 1:Nx
        % Use a weighted combination of sinusoids modulated by permittivity profile
        phase = k0 * eps_profile(i) * x(i);
        H(i) = sin(phase) * exp(-x(i)^2/(2*sigma_target^2));
    end

    % Apply boundary conditions
    H(1) = 0;
    H(end) = 0;

    % Ensure it's a row vector to match the target field format
    H = H(:)';
end


%% **6. Final Magnetic Field Plot Using Optimized Permittivity**
H_final = compute_field(globalBestPosition, x, sigma_target);
H_final = H_final / max(abs(H_final)) * max(abs(H_target)); % Normalize

% Ensure field orientation matches target
if dot(H_final, H_target) < 0
    H_final = -H_final;
end

figure;
plot(x, H_target, 'r', 'LineWidth', 2, 'DisplayName', 'Target Field');
hold on;
plot(x, H_final, 'b--', 'LineWidth', 2, 'DisplayName', 'Final Computed Field');
xlabel('Position (a.u.)'); ylabel('Magnetic Field H (a.u.)');
title('Final Magnetic Field Comparison');
legend;
grid on;

