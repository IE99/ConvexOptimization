% A function to determine how a photonic structure may look like after
% fabrication
% Inputs:
%   permittivity: An array of permittivity values
%   N: Number of points in the structure
% Outputs:
%  filtered_permittivity: An array of permittivity values, after filtering
function filtered_permittivity = Filter(permittivity, N)
    min_feature_size = 0.5; % Minimum feature size, assuming length of domain is 20
    dx = 20/N; % Distance between points, assuming length of domain is 20

    sigma = min_feature_size / dx; % Convert physical size to grid units
    filter_size = ceil(6 * sigma); % Ensure the filter captures enough of the Gaussian
    g_filter = fspecial('gaussian', [1, filter_size], min_feature_size); % 1D Gaussian filter

    filtered_permittivity = conv(permittivity, g_filter, 'same');
end