
clearvars; close all; clc;
disp(pwd);
filename = 'data\RGB2CIELAB_1';
x0 = 1;

% Load mesh
Mm = MeshClass(filename);

%% Regularized - Dirichlet Energy
alpha_hat0 = 0.05;

% Initial computation
u_D1 = rdg_ADMM(Mm, x0, 'alpha_hat', alpha_hat0);
writematrix(u_D1, 'u_D1.csv');

num_iterations = Mm.nv;
disp(num_iterations)
max_indices = zeros(1, num_iterations);


% Start parallel pool (if not already started)
% if isempty(gcp('nocreate'))
%     parpool;  % Uses default number of workers
% end

% Parallel loop
for i = 1:num_iterations
    % Find index of largest value
    u = rdg_ADMM(Mm, i, 'alpha_hat', alpha_hat0);
    [~, max_idx] = max(u);
    
    % Store result
    max_indices(i) = max_idx;

    disp(i);
    
    % Note: disp() in parfor may appear out of order
    % fprintf('Iteration %d complete\n', i);
    %  if mod(i, 100) == 0  % Print every 100 iterations
    %     fprintf('Progress: %d/%d\n', i, num_iterations);
    % end
end

% Write to file at the end
writematrix(max_indices,'max_indices_alpha_hat_05.txt');


% % given directions:

% alpha_hat = 0.05;
% beta_hat = 100;
% 
% u_vfa = rdg_ADMM(Mm, x0, 'reg', 'vfa', 'alpha_hat', alpha_hat, 'beta_hat', beta_hat, 'vf', vf_int);
% 
% 
% 
% %% Figures
% cam = load('spot_rr_cam.mat'); cam = cam.cam;
% u_all = [u0(:); u_D1(:); u_D2(:); u_vfa(:)];
% umin = min(u_all);
% umax = max(u_all);
% nlines = 15;
% 
% Mm.visualizeDistances(u0, x0, nlines, [umin, umax], cam);
% Mm.visualizeDistances(u_D1, x0, nlines, [umin, umax], cam);
% Mm.visualizeDistances(u_D2, x0, nlines, [umin, umax], cam);
% 
% Mm.visualizeDistances(u_vfa, x0, nlines, [umin, umax], cam); hold on;
% br = Mm.baryCentersCalc;
% quiver3(br(:,1),br(:,2),br(:,3),...
%     vf(:,1),vf(:,2),vf(:,3),2,'color','k','LineWidth',2,'ShowArrowHead','off');
% quiver3(br(:,1),br(:,2),br(:,3),...
%     -vf(:,1),-vf(:,2),-vf(:,3),2,'color','k','LineWidth',2,'ShowArrowHead','off');
% 
% 
% 
% 
