%% File names for all seven bands
file_band1 = 'muneebB1.TIF';
file_band2 = 'muneebB2.TIF';
file_band3 = 'muneebB3.TIF';
file_band4 = 'muneebB4.TIF';
file_band5 = 'muneebB5.TIF';
file_band6 = 'muneebB6.TIF';
file_band7 = 'muneebB7.TIF';

% Read all bands
band1 = imread(file_band1);
band2 = imread(file_band2);
band3 = imread(file_band3);
band4 = imread(file_band4);
band5 = imread(file_band5);
band6 = imread(file_band6);
band7 = imread(file_band7);

% Define crop size (height and width of the crop area)
crop_height = 250; % Desired height of the cropped area
crop_width = 250;  % Desired width of the cropped area

% Function to crop the center of a band
crop_center = @(band) band( ...
    round(size(band, 1) / 2 - crop_height / 2):round(size(band, 1) / 2 + crop_height / 2 - 1), ...
    round(size(band, 2) / 2 - crop_width / 2):round(size(band, 2) / 2 + crop_width / 2 - 1));

% Crop the center part of each band
cropped_band1 = crop_center(band1);
cropped_band2 = crop_center(band2);
cropped_band3 = crop_center(band3);
cropped_band4 = crop_center(band4);
cropped_band5 = crop_center(band5);
cropped_band6 = crop_center(band6);
cropped_band7 = crop_center(band7);

% Display the cropped bands
figure;
subplot(3, 3, 1);
imshow(cropped_band1, []);
title('Cropped Band 1');

subplot(3, 3, 2);
imshow(cropped_band2, []);
title('Cropped Band 2');

subplot(3, 3, 3);
imshow(cropped_band3, []);
title('Cropped Band 3');

subplot(3, 3, 4);
imshow(cropped_band4, []);
title('Cropped Band 4');

subplot(3, 3, 5);
imshow(cropped_band5, []);
title('Cropped Band 5');

subplot(3, 3, 6);
imshow(cropped_band6, []);
title('Cropped Band 6');

subplot(3, 3, 7);
imshow(cropped_band7, []);
title('Cropped Band 7');

% Concatenate the bands
group1 = cat(3, cropped_band1, cropped_band2);
group2 = cat(3, cropped_band3, cropped_band4);
group3 = cat(3, cropped_band5, cropped_band6, cropped_band7);
final_image = cat(3, group1, group2, group3);

% Normalize the final concatenated image for visualization
normalized_final_image = double(final_image) / double(max(final_image(:)));

% Create a False-Color RGB composite (Bands 1, 4, 7 as an example)
rgb_composite = cat(3, normalized_final_image(:, :, 1), ... % Red (Band 1)
                       normalized_final_image(:, :, 4), ... % Green (Band 4)
                       normalized_final_image(:, :, 7));   % Blue (Band 7)

% Display the RGB composite
figure;
imshow(rgb_composite);
title('Original RGB Composite (Bands 1, 4, 7)');

% Reshape the RGB composite for PCA
[m, n, c] = size(rgb_composite);
data_matrix = double(reshape(rgb_composite, [], c)); % Pixels x channels
mean_vector = mean(data_matrix, 1);
data_centered = data_matrix - mean_vector;

% Calculate the covariance matrix and eigenvalues/eigenvectors
cov_matrix = (data_centered' * data_centered) / size(data_centered, 1);
[eigenvectors, eigenvalues_matrix] = eig(cov_matrix);
eigenvalues = diag(eigenvalues_matrix);
[~, indices] = sort(eigenvalues, 'descend');
eigenvalues = eigenvalues(indices);
eigenvectors = eigenvectors(:, indices);

% Cumulative variance explained
total_variance = sum(eigenvalues);
explained_variance = eigenvalues / total_variance;
cumulative_variance = cumsum(explained_variance);

% Reconstruction error vs number of PCs
max_k = size(eigenvectors, 2); % Maximum number of PCs
errors = zeros(max_k, 1);
for k = 1:max_k
    eigenvectors_k = eigenvectors(:, 1:k);
    projected_data = data_centered * eigenvectors_k;
    reconstructed_data = projected_data * eigenvectors_k' + mean_vector;
    errors(k) = norm(data_matrix - reconstructed_data, 'fro') / norm(data_matrix, 'fro'); % Frobenius norm
end

% Display plots for reconstruction error and cumulative variance
figure;
subplot(1, 2, 1);
plot(1:max_k, errors, '-o', 'LineWidth', 1.5);
xlabel('Number of Principal Components (k)');
ylabel('Reconstruction Error');
title('Reconstruction Error vs Number of PCs');
grid on;

subplot(1, 2, 2);
plot(1:max_k, cumulative_variance, '-o', 'LineWidth', 1.5);
xlabel('Number of Principal Components (k)');
ylabel('Cumulative Variance Explained');
title('Cumulative Variance vs Number of PCs');
grid on;

% Reconstruct the image using top 2 PCs
top_k = 2;
eigenvectors_2 = eigenvectors(:, 1:top_k);
projected_data_2 = data_centered * eigenvectors_2;
reconstructed_data_2 = projected_data_2 * eigenvectors_2' + mean_vector;
reconstructed_image_2 = reshape(reconstructed_data_2, m, n, c);

% Extract the second PC and reshape it for visualization
second_pc = projected_data_2(:, 2);
second_pc_image = reshape(second_pc, m, n);

% Display the original RGB composite, reconstructed image, and second PC image
figure;

subplot(1, 3, 1);
imshow(rgb_composite);
title('Original RGB Image');

subplot(1, 3, 2);
imshow(uint8(reconstructed_image_2));
title('Reconstructed Image using 2 PCs');

subplot(1, 3, 3);
imshow(second_pc_image, []);
title('Second Principal Component (PC2)');
