function [w, cluster] = L0_kmeans(X, ncl, TopK, maxiter)
    if nargin < 3 
        TopK = 20;
    end
    if nargin < 4
        maxiter = 5;
    end
    if TopK>size(X, 1)
        TopK=size(X, 1);
    end
    % Initialize w0 with equal weights on each feature
    XT=X';
    w0 = ones(1, size(X, 1));
    for kk = 1:maxiter
        % Update data matrix X_
        X1 = XT(:, w0 == 1);
        km = kmeans(X1, ncl); 
%         km = kmeans(X1, ncl); 
        cluster = km;
        % Calculate the 'a' vector
        a = calculate_a_vector(XT, cluster);
        % Update w vector
        w = zeros(size(w0));
        % Sort 'a' in descending order and set the top TopK values in w to 1
        [~, idx] = sort(a, 'descend');
        w(idx(1:TopK)) = 1;
        % Convergence check
        if sum(abs(w - w0)) / sum(abs(w0)) < 1e-4
            break;
        end
        w0 = w;
    end
end


function a = calculate_a_vector(X, cluster)
    % Number of samples
    n = size(X, 1);
    % Number of features
    p = size(X, 2);
    % Initialize 'a' vector with zeros
    a = zeros(1, p);
    
    % Calculate pairwise distances for the whole dataset
    for j = 1:p
        X_j = X(:, j);  % j-th feature of X
        temp=pdist2(X_j, X_j);
        sum1 = sum(temp(:)) / n / 2;  
        sum2 = 0;

        uc=unique(cluster);
        for k = 1:length(uc)
            cluster_id = uc(k);
            % Members in cluster k
            id = find(cluster == cluster_id);
            n_k = length(id);
            X_j_k = X_j(id);
            temp=pdist2(X_j_k, X_j_k);
            sum2 = sum2 + sum(temp(:)) / n_k / 2;
        end
        a(j) = sum1 - sum2;
    end
end