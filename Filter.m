permittivity = epsilon;
min_feature_size = 3;
dx = 1;

sigma = min_feature_size / dx;  % Convert physical size to grid units
filter_size = ceil(6 * sigma); % Ensure the filter captures enough of the Gaussian
g_filter = fspecial('gaussian', [1, filter_size], sigma); % 1D Gaussian filter

permittivity_smoothed = conv(permittivity, g_filter, 'same');

% threshold = 5;
% bw = permittivity > threshold; % Convert to binary structure
% bw = bwareaopen(bw, round(min_feature_size / dx)); % Remove small features
% 
% permittivity_smoothed = bw * (max(permittivity) - min(permittivity)) + min(permittivity);


subplot(2, 1, 1);
plot(permittivity, 'r');
subplot(2, 1, 2);
plot(permittivity_smoothed, 'b');
title('Permittivity Profile with Gaussian Smoothing');