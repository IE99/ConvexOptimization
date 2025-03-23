clc; clear; close all;

Nx = 500;
x  = linspace(-10, 10, Nx)';

h1 = heaviside(x + 5);  
h2 = heaviside(x - 5);  
box_fn = h1 - h2;
H_target = box_fn;

eps_min = 1;
eps_max = 10;
numParticles  = 500;    
numDimensions = Nx;      
maxIterations = 5000;    

eps_mid      = (eps_min + eps_max)/2;
sigma_eps    = 3;
base_profile = eps_mid * ones(1, Nx);

%rng('shuffle');
positions  = zeros(numParticles, numDimensions);
velocities = zeros(numParticles, numDimensions);

for i = 1:numParticles
    if i == 1
        positions(i,:) = base_profile;
    else
        variation = sigma_eps * (rand(1, numDimensions) - 0.5);
        trial     = base_profile + variation;
        positions(i,:) = max(eps_min, min(eps_max, trial));
    end
    velocities(i,:) = 0.05 * (rand(1, numDimensions) - 0.5);
end

w = 0.7; c1 = 1.5; c2 = 1.5;

personalBestPositions = positions;
personalBestScores    = zeros(numParticles, 1);

for i = 1:numParticles
    personalBestScores(i) = compute_fitness(positions(i,:), x, H_target, eps_min, eps_max);
end

[globalBestScore, gIdx]  = min(personalBestScores);
globalBestPosition       = personalBestPositions(gIdx,:);
bestFitnessHistory = zeros(maxIterations, 1);

for iter = 1:maxIterations
    for i = 1:numParticles
        r1 = rand(1, numDimensions);
        r2 = rand(1, numDimensions);
        velocities(i,:) = w * velocities(i,:) ...
                        + c1 * r1 .* ( personalBestPositions(i,:) - positions(i,:) ) ...
                        + c2 * r2 .* ( globalBestPosition - positions(i,:) );

        positions(i,:) = positions(i,:) + velocities(i,:);
        positions(i,:) = max(eps_min, min(eps_max, positions(i,:)));

        newFitness = compute_fitness(positions(i,:), x, H_target, eps_min, eps_max);
        if newFitness < personalBestScores(i)
            personalBestScores(i)    = newFitness;
            personalBestPositions(i,:) = positions(i,:);
        end
    end

    [thisBestScore, bIdx] = min(personalBestScores);
    if thisBestScore < globalBestScore
        globalBestScore   = thisBestScore;
        globalBestPosition = personalBestPositions(bIdx,:);
    end

    bestFitnessHistory(iter) = globalBestScore;
end

filtered_eps = Filter(globalBestPosition, Nx);

H_optimized = compute_field(globalBestPosition, x, eps_min, eps_max);
H_filtered  = compute_field(filtered_eps, x, eps_min, eps_max);

H_optimized = H_optimized / max(abs(H_optimized)) * max(abs(H_target));
H_filtered  = H_filtered  / max(abs(H_filtered))  * max(abs(H_target));
H_optimized = H_optimized - mean(H_optimized) + mean(H_target);
H_filtered  = H_filtered  - mean(H_filtered)  + mean(H_target);

err_unfiltered = norm(H_optimized - H_target);
err_filtered   = norm(H_filtered  - H_target);
err_filter_effect = norm(H_optimized - H_filtered);

fprintf('Error (Unfiltered Field vs Target): %.6f\n', err_unfiltered);
fprintf('Error (Filtered Field vs Target):   %.6f\n', err_filtered);
fprintf('Change due to Filtering:            %.6f\n', err_filter_effect);

figure('Name','Final Results','Color','w');

subplot(2,1,1);
plot(x, H_target,    'r',  'LineWidth', 2, 'DisplayName', 'Target Field'); hold on;
plot(x, H_optimized, 'b--','LineWidth', 2, 'DisplayName', 'Optimized Field');
plot(x, H_filtered,  'm-.','LineWidth', 2, 'DisplayName', 'Filtered Field');
xlabel('x'); ylabel('Magnetic Field H(x)');
title('Target vs Optimized vs Filtered Field');
legend('Location','Best'); grid on;

subplot(2,1,2);
plot(x, globalBestPosition, 'g',  'LineWidth', 2, 'DisplayName','Optimized \epsilon(x)'); hold on;
plot(x, filtered_eps,       'k--','LineWidth', 2, 'DisplayName','Filtered \epsilon(x)');
xlabel('x'); ylabel('Dielectric Permittivity \epsilon(x)');
title('Permittivity Profiles');
legend('Location','Best'); grid on;

fprintf('===== Final Results =====\n');
disp('Optimized Permittivity Profile (first 10 values):');
disp(globalBestPosition(1:10));
fprintf('Final Best Fitness: %.6f\n', globalBestScore);


function cost = compute_fitness(eps_profile, x, H_target, eps_min, eps_max)
    H_computed = compute_field(eps_profile, x, eps_min, eps_max);
    cost = norm(H_computed - H_target);
end

function H = compute_field(eps_profile, x, eps_min, eps_max)
    Nx = length(x);
    dx = x(2) - x(1);
    avgEps = (eps_min + eps_max)/2;
    H = zeros(Nx,1);
    for i = 2:Nx
        H(i) = H(i-1) + ( eps_profile(i) - avgEps ) * dx;
    end
end

function filtered_permittivity = Filter(permittivity, N)
    min_feature_size = 0.5; 
    dx = 20 / N;
    sigma = min_feature_size / dx;
    filter_size = ceil(6 * sigma);
    if mod(filter_size, 2) == 0
        filter_size = filter_size + 1;
    end
    half = floor(filter_size/2);
    x_filter = -half:half;
    g_filter = exp(-x_filter.^2 / (2 * sigma^2));
    g_filter = g_filter / sum(g_filter);
    filtered_permittivity = conv(permittivity, g_filter, 'same');
end
