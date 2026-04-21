
clearvars; close all; clc;
disp(pwd);
filename = 'RGB2CIELAB_neural_1';
final_file = 'max_indices_cielab_1_RGD_125_neural_256.txt';
x0 = 1;

% Load mesh
Mm = MeshClass(filename);

%% Regularized - Dirichlet Energy
alpha_hat0 = 1.25;

% Initial computation
u_D1 = rdg_ADMM(Mm, x0, 'alpha_hat', alpha_hat0);
writematrix(u_D1, 'u_D1.csv');

num_iterations = Mm.nv;
disp(num_iterations)
max_indices = zeros(1, num_iterations);

if isempty(gcp('nocreate'))
    parpool;  
end

parfor i = 1:num_iterations
    u = rdg_ADMM(Mm, i, 'alpha_hat', alpha_hat0);
    [~, max_idx] = max(u);
    
    max_indices(i) = max_idx;
    if mod(i,1000) == 0
        disp(i);
    end
end

writematrix(max_indices, final_file);
disp(final_file);