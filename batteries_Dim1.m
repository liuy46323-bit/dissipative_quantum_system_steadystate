clc; clear;

%% 参数设置
Dim = 64;       % 系统Hilbert空间维度
eta = 1;        % 未使用的参数，可能是耦合强度
gamma = 1;      % Lindblad耗散强度
I = eye(Dim);   % Dim维单位矩阵

%% 哈密顿量H构造（周期边界条件的1D链）
H = zeros(Dim);
for k = 2:Dim-1
    H(k,k-1) = 1;
    H(k,k+1) = 1;
end
H(1,Dim) = 1; H(1,2) = 1;          % 周期边界条件
H(Dim,1) = 1; H(Dim,Dim-1) = 1;

%% H对角化得到本征态和本征能量
[V,D] = eig(H);
E = diag(D);
[E_sort, idx] = sort(E);            % 将能量升序排序
V_sort = V(:, idx);                 % 对应的能量本征态排序

%% Lindblad耗散算符矩阵构造（超算符形式）
D_T = zeros(Dim^2);
for k = 1:Dim
    L = zeros(Dim);
    i = k;
    if i == Dim
        j = 1;
    else
        j = i+1;
    end
    % 构造每个局域Lindblad算符L_{k}
    L(i,i) = 1;
    L(j,i) = 1;
    L(i,j) = -1;
    L(j,j) = -1;
    
    % Lindblad超算符贡献 (kron表示向量化形式)
    D_T = D_T + gamma*(kron(conj(L),L) - 0.5*(kron(L'*L, I) + kron(I, (L'*L).')));
end

%% Liouvillian算符
L_H = -1i*(kron(I,H) - kron(H.',I));  % 哈密顿部分
Lindblad = L_H + D_T;                 % 总Liouvillian

%% 求稳态密度矩阵（Liouvillian的零本征值对应稳态）
[Vec,Val] = eig(Lindblad);
eigvals = diag(Val);
[~, idx0] = min(abs(eigvals));        % 找到最接近零的特征值
rho_ss_vec = Vec(:, idx0);            % 对应特征向量
rho_ss = reshape(rho_ss_vec, Dim, Dim);
rho_ss = (rho_ss + rho_ss') / 2;      % 强制Hermitian
rho_ss = rho_ss / trace(rho_ss);      % 归一化

%% 稳态密度矩阵本征值分析
[V1,D1] = eig(rho_ss);
p = diag(D1);
p = real(p);                         % 数值计算可能有微小虚部
[p_sort, idx1] = sort(p,'descend');  % 降序排列

%% 稳态平均能量和被动能量计算
E_avg = trace(H*rho_ss);              % 平均能量
E_avg = real(E_avg);
E_passive = sum(p_sort(:).' * E_sort(:));   % 被动能量
E_passive = real(E_passive);
ep_ss = E_avg - E_passive;            % 稳态可提取能量

%% 时间演化准备
t_list = 0:0.2:50;                    % 时间点列表
rho0 = zeros(Dim);                    % 初态构造（按稳态本征值分布）
for k = 1:Dim
    psi = V_sort(:,k);
    rho0 = rho0 + p_sort(k)*(psi*psi');
end
rho0_vec = reshape(rho0, Dim^2, 1);  % 向量化

N = length(t_list);
rhot_vec_list = zeros(Dim^2,N);
ep_t_list = zeros(1,N);

% 预计算系数：用特征分解代替每次求解
coeff = Vec \ rho0_vec;

%% 时间演化计算
for k = 1:N
    rhot_vec = Vec * (exp(eigvals * t_list(k)) .* coeff);  % Liouvillian对角化求解
    rhot = reshape(rhot_vec, Dim, Dim);
    rhot = (rhot + rhot') / 2;      % Hermitian化
    rhot = rhot / trace(rhot);      % 归一化

    [~, D2] = eig(rhot);
    pt_sort = sort(real(diag(D2)), 'descend');

    Et_passive = sum(pt_sort(:).*E_sort(:));
    ep_t_list(k) = real(trace(H * rhot)) - Et_passive;   % 时间演化可提取能量
end

%% Figure 1(b): 稳态密度矩阵在能量本征态基下（高能部分）
rho_energy = V_sort' * rho_ss * V_sort; 
rho_energy = real(rho_energy);
n_range = 50:Dim;

figure(1)
imagesc(n_range, n_range, rho_energy(n_range, n_range))
colormap(jet)
colorbar
xlabel('n')
ylabel('n')
title('Steady-state density matrix in energy basis (high-n region)')
set(gca, 'FontSize', 14)
axis square
set(gca, 'YDir', 'normal')

%% Figure 1(c): 可提取能量随时间演化
figure(2)
plot(gamma*t_list, ep_t_list, 'LineWidth', 2)
hold on
h = yline(ep_ss, '--', '\epsilon_{ss}', ...
    'LineWidth', 2, ...
    'Color', 'k');        % 标出稳态值

h.Label = '\epsilon_{ss}';
h.FontSize = 18;

hold off
xlabel('\gamma t')
ylabel('\epsilon(\tau)')
xlim([0, max(gamma*t_list)])
ylim([0, 5])
set(gca, 'FontSize', 14)
grid on