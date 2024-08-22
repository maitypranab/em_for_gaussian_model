
% Step 1: Generate synthetic dataset
mu1 = [3, 3];
sigma1 = [1, 0; 0, 2];
pi1 = 0.8;

mu2 = [1, -3];
sigma2 = [2, 0; 0, 1];
pi2 = 0.2;

N = 500;

% Generate data points
n1 = round(pi1 * N);
data1 = mvnrnd(mu1, sigma1, n1);
n2 = N - n1;
data2 = mvnrnd(mu2, sigma2, n2);
data = [data1; data2];

%plot the synthetic data set
figure;
scatter(data1(:, 1),data1(:, 2), 'r','filled');
hold on
scatter(data2(:, 1),data2(:, 2),'b','filled');
hold off
title('Synthetic Dataset');
legend('cluster 1','cluster 2',Location='best');

%Randomize the synthetic data set
data=data(randperm(size(data, 1)), :);

% Step 2: EM algorithm
K = 2;

% Initialize parameters
pi = ones(1, K) / K;  % Random initialization
mu = rand(K, 2);  % Random initialization
sigma = repmat(eye(2), [1, 1, K]);  % Identity matrix for each cluster

% Initialize assignment probabilities
q = zeros(N, K);

% EM algorithm
max_iter = 100;
tolerance = 1e-6;
prev_lower_bound = -inf;

for iter = 1:max_iter
    % Step 1: E-step
    for k = 1:K
        q(:, k) = pi(k) * mvnpdf(data, mu(k, :), squeeze(sigma(:, :, k)));
    end
    q = q ./ sum(q, 2);
    
    % Step 2: M-step
    for k = 1:K
        Nk = sum(q(:, k));
        pi(k) = Nk / N;
        mu(k, :) = sum(data .* q(:, k), 1) / Nk;
        diff = data - mu(k, :);
        sigma(:, :, k) = (diff' * (diff .* q(:, k))) / Nk;
    end
    
    % Calculate lower bound
    log_likelihood = 0;
    for k = 1:K
        log_likelihood = log_likelihood + sum(log(pi(k) * mvnpdf(data, mu(k, :), sigma(:, :, k))));
    end
    lower_bound = log_likelihood + sum(q .* log(q));
    
    % Check for convergence
    if abs(lower_bound - prev_lower_bound) < tolerance
        break;
    end
    
    prev_lower_bound = lower_bound;
end

% Assign points to clusters
[~, labels] = max(q, [], 2);

%plot the updated dataset
figure;
scatter(data(labels==1, 1), data(labels==1, 2), 'r','filled');
hold on;
scatter(data(labels==2, 1), data(labels==2, 2), 'b','filled');
hold off
title('Updated Dataset');
legend('cluster 1','cluster 2',Location='best');

% Print results
disp('Final parameters after convergence:');
disp('pi:');
disp(pi);
disp('mu:');
disp(mu);
disp('sigma:');
disp(sigma);
