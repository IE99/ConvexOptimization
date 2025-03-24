clc; clear; close all;

Nx = 500; 
x = linspace(-10, 10, Nx);

A_target = 1;
sigma_target = 2;
k_target = 5;
phi_target = 0;
H_target = A_target * exp(-x.^2 / (2 * sigma_target^2)) .* cos(k_target*x + phi_target);
H_target = H_target(:); 

eps_min = 1; % Minimum permittivity
eps_max = 10; % Maximum permittivity
numParticles = 500;
numDimensions = Nx; 
maxIterations = 5000;

% Gaussian base profile
eps_background = eps_min;
eps_center = (eps_min + eps_max) / 2;
sigma_eps = 4;

base_profile = eps_background + (eps_center - eps_background) * exp(-x.^2 / (2 * sigma_eps^2));

positions = zeros(numParticles, numDimensions);
for i = 1:numParticles
    if i == 1
        positions(i, :) = base_profile;
    else
        variation = 0.2 * (eps_max - eps_min) * (rand(1, numDimensions) - 0.5);
        positions(i, :) = base_profile + variation;
        positions(i, :) = max(eps_min, min(eps_max, positions(i, :)));
    end
end

velocities = -0.05 + 0.1 * rand(numParticles, numDimensions);

w = 0.7; c1 = 1.5; c2 = 1.5;

personalBestPositions = positions;
personalBestScores = zeros(numParticles, 1);
for i = 1:numParticles
    personalBestScores(i) = compute_fitness(positions(i, :), x, H_target, sigma_target);
end
[globalBestScore, bestIndex] = min(personalBestScores);
globalBestPosition = personalBestPositions(bestIndex, :);

bestFitnessHistory = zeros(maxIterations, 1);

for iter = 1:maxIterations
    for i = 1:numParticles
        r1 = rand(1, numDimensions);
        r2 = rand(1, numDimensions);
        velocities(i, :) = w * velocities(i, :) ...
                         + c1 * r1 .* (personalBestPositions(i, :) - positions(i, :)) ...
                         + c2 * r2 .* (globalBestPosition - positions(i, :));

        positions(i, :) = positions(i, :) + velocities(i, :);
        positions(i, :) = max(eps_min, min(eps_max, positions(i, :)));

        newFitness = compute_fitness(positions(i, :), x, H_target, sigma_target);
        if newFitness < personalBestScores(i)
            personalBestScores(i) = newFitness;
            personalBestPositions(i, :) = positions(i, :);
        end
    end

    [currentBestScore, bestIndex] = min(personalBestScores);
    if currentBestScore < globalBestScore
        globalBestScore = currentBestScore;
        globalBestPosition = personalBestPositions(bestIndex, :);
    end
    bestFitnessHistory(iter) = globalBestScore;
end

H_optimized = compute_field(globalBestPosition, x, sigma_target);
H_optimized = H_optimized(:);
H_optimized = H_optimized / max(abs(H_optimized)) * max(abs(H_target));
if dot(H_optimized, H_target) < 0
    H_optimized = -H_optimized;
end

filtered_eps = Filter(globalBestPosition, Nx);
H_filtered = compute_field(filtered_eps, x, sigma_target);
H_filtered = H_filtered(:);
H_filtered = H_filtered / max(abs(H_filtered)) * max(abs(H_target));
if dot(H_filtered, H_target) < 0
    H_filtered = -H_filtered;
end

error_filtered = norm(H_filtered - H_target);

normalized_error_filtered = error_filtered / norm(H_target);

fprintf('Error in filtered field (L2 norm): %.6f\n', error_filtered);
fprintf('Normalized error in filtered field: %.6f\n', normalized_error_filtered);




figure;

subplot(2,1,1);
plot(x, H_target, 'r', 'LineWidth', 2, 'DisplayName', 'Target Field');
hold on;
plot(x, H_optimized, 'b--', 'LineWidth', 2, 'DisplayName', 'Optimized Field');
plot(x, H_filtered, 'm-.', 'LineWidth', 2, 'DisplayName', 'Filtered Field');
xlabel('Position (a.u.)'); ylabel('Magnetic Field H (a.u.)');
title('Magnetic Field Comparison');
legend('Location', 'Best');
grid on;

subplot(2,1,2);
plot(x, globalBestPosition, 'g', 'LineWidth', 2, 'DisplayName', 'Optimized Permittivity');
hold on;
plot(x, filtered_eps, 'k--', 'LineWidth', 2, 'DisplayName', 'Filtered Permittivity');
xlabel('Position (a.u.)'); ylabel('Dielectric Permittivity (\epsilon)');
title('Permittivity Profiles');
legend('Location', 'Best');
grid on;

figure;
plot(bestFitnessHistory, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Fitness Value');
title('PSO Convergence');
grid on;

disp('Optimized Permittivity Profile:');
disp(globalBestPosition);
fprintf('Final Best Fitness: %.6f\n', globalBestScore);

function cost = compute_fitness(eps_profile, x, H_target, sigma_target)
    H_computed = compute_field(eps_profile, x, sigma_target);
    H_computed = H_computed(:);
    H_target = H_target(:);
    H_computed = H_computed / max(abs(H_computed)) * max(abs(H_target));
    if dot(H_computed, H_target) < 0
        H_computed = -H_computed;
    end
    cost = norm(H_computed - H_target);
end

function H = compute_field(eps_profile, x, sigma_target)
    dx = x(2) - x(1);
    Nx = length(x);
    k0 = 5;
    H = zeros(size(x));
    for i = 1:Nx
        phase = k0 * eps_profile(i) * x(i);
        H(i) = sin(phase) * exp(-x(i)^2/(2*sigma_target^2));
    end
    H(1) = 0;
    H(end) = 0;
    H = H(:)';
end

function filtered_permittivity = Filter(permittivity, N)
    min_feature_size = 0.5; 
    dx = 20/N;
    sigma = min_feature_size / dx; 
    filter_size = ceil(6 * sigma); 
    g_filter = fspecial('gaussian', [1, filter_size], min_feature_size); 
    filtered_permittivity = conv(permittivity, g_filter, 'same');
end



