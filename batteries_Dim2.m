clc;clear;

%% ===================== 基本参数设置 =====================
N0 = 10;                     % 2D晶格每一维的格点数
Dim = N0^2;                  % 总希尔伯特空间维度（2D展开为1D索引）
eta = 1;                     % 参数（当前未显式使用）
gamma = 1;                   % 耗散强度 / Lindblad系数
I = eye(Dim);                % 单位矩阵（用于Liouvillian构造）

X = 1:1:N0;
Y = 1:1:N0;

%% ===================== 哈密顿量与耗散超算符初始化 =====================
H = zeros(Dim);              % 系统哈密顿量（tight-binding结构）
D_T = zeros(Dim^2);         % Lindblad耗散项对应的超算符

%% ===================== 构造晶格哈密顿量 + 跳跃算符 =====================
for k = 1:Dim
    
    % 每个格点对应三个Lindblad跳跃通道（L1, L2, L3）
    L1 = zeros(Dim);
    L2 = zeros(Dim);
    L3 = zeros(Dim);

    % 将1D索引映射回2D坐标 (x,y)
    x = mod(k-1,N0)+1;
    y = floor((k-1)/N0)+1;

    % -------- 周期边界条件下的邻居索引 --------
    x1 = x;                 % x方向不变
    y1 = mod(y,N0)+1;       % y+1（周期）
    
    x2 = mod(x-2,N0)+1;     % x-1（周期）
    y2 = mod(y,N0)+1;

    x3 = x;
    y3 = mod(y-2,N0)+1;     % y-1（周期）

    % 只在“奇数行”构造跳跃（体现结构性耦合/非对称驱动）
    if mod(y,2) == 1
        
        i = k;
        j1 = (y1-1)*N0+x1;
        j2 = (y2-1)*N0+x2;
        j3 = (y3-1)*N0+x3;

        %% ===================== 构造哈密顿量（无耗散部分） =====================
        % 三个方向的最近邻跃迁（对称tight-binding）
        H(i,j1) = 1; H(i,j2) = 1; H(i,j3) = 1;
        H(j1,i) = 1; H(j2,i) = 1; H(j3,i) = 1;

        %% ===================== 构造Lindblad跳跃算符 =====================
        % L1 / L2 / L3 对应三个方向的局域跃迁耗散通道
        L1(i,i) = 1; L1(j1,i) = 1;
        L1(i,j1) = -1; L1(j1,j1) = -1;

        L2(i,i) = 1; L2(j2,i) = 1;
        L2(i,j2) = -1; L2(j2,j2) = -1;

        L3(i,i) = 1; L3(j3,i) = 1;
        L3(i,j3) = -1; L3(j3,j3) = -1;

        %% ===================== Lindblad耗散超算符构造 =====================
        % 标准形式：
        % D[ρ] = LρL† - 1/2 {L†L, ρ}
        D_T = D_T + gamma*(kron(conj(L1),L1) ...
            - 0.5*(kron(L1'*L1, I) + kron(I, (L1'*L1).'))) ...
            + gamma*(kron(conj(L2),L2) ...
            - 0.5*(kron(L2'*L2, I) + kron(I, (L2'*L2).'))) ...
            + gamma*(kron(conj(L3),L3) ...
            - 0.5*(kron(L3'*L3, I) + kron(I, (L3'*L3).')));

    else
        % 偶数行不施加该结构性耗散（形成“分层晶格结构”）
    end
end

%% ===================== 构造Liouvillian超算符 =====================
L_H = -1i*(kron(I,H) - kron(H.',I));   % 冯诺依曼方程对应的哈密顿部分
Lindblad = L_H + D_T;                  % 总Liouvillian

%% ===================== 哈密顿量本征态分析 =====================
[V,D] = eig(H);
E = diag(D);

% 能量排序（用于passive state构造）
[E_sort, idx] = sort(E);
V_sort = V(:, idx);

%% ===================== 稳态密度矩阵（Liouvillian零模） =====================
[Vec,Val] = eig(Lindblad);
eigvals = diag(Val);

% 找最接近0的本征值 → 稳态
[~, idx0] = min(abs(eigvals));
rho_ss_vec = Vec(:, idx0);

% reshape回密度矩阵
rho_ss = reshape(rho_ss_vec, Dim, Dim);

% 数值修正：保证Hermitian + 归一化
rho_ss = (rho_ss + rho_ss') / 2;
rho_ss = rho_ss / trace(rho_ss);

%% ===================== 稳态本征值（用于passive energy） =====================
[V1,D1] = eig(rho_ss);
p = real(diag(D1));

% 降序排列（用于构造passive state）
[p_sort, idx1] = sort(p,'descend');

%% ===================== 能量与被动能计算 =====================
E_avg = real(trace(H*rho_ss));          % 平均能量

% passive energy：重新排列占据数，使其与能量升序匹配
E_passive = real(sum(p_sort(:).' * E_sort(:)));

% 可提取能量（ergotropy）
ep_ss = E_avg - E_passive;

%% ===================== 时间演化初始化 =====================
t_list = 0:0.25:80;                    % 时间采样点

% 构造初态：按稳态本征态分布构造“被动初态”
rho0 = zeros(Dim);

for k = 1:Dim
    psi = V_sort(:,k);
    rho0 = rho0 + p_sort(k)*(psi*psi');
end

rho0_vec = reshape(rho0, Dim^2, 1);

N = length(t_list);
rhot_vec_list = zeros(Dim^2,N);
ep_t_list = zeros(1,N);

% Liouvillian对角化解时间演化系数
coeff = Vec \ rho0_vec;

%% ===================== 时间演化主循环 =====================
for k = 1:N

    % 解析解：rho(t) = exp(L t) rho(0)
    rhot_vec = Vec * (exp(eigvals * t_list(k)) .* coeff);

    rhot = reshape(rhot_vec, Dim, Dim);

    % 数值稳定处理
    rhot = (rhot + rhot') / 2;
    rhot = rhot / trace(rhot);

    % instantaneous passive energy
    [~, D2] = eig(rhot);
    pt_sort = sort(real(diag(D2)), 'descend');

    Et_passive = sum(pt_sort(:).*E_sort(:));

    % 可提取能量随时间变化
    ep_t_list(k) = real(trace(H * rhot)) - Et_passive;
end

%% ===================== Figure 1(b): 能量本征基下稳态密度矩阵 =====================
rho_energy = V_sort' * rho_ss * V_sort;
rho_energy = real(rho_energy);

n_range = 90:Dim;

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

%% ===================== Figure 1(c): 可提取能量时间演化 =====================
figure(2)
plot(gamma*t_list, ep_t_list, 'LineWidth', 2)
hold on

% 稳态可提取能量参考线
h = yline(ep_ss, '--', '\epsilon_{ss}', ...
    'LineWidth', 2, ...
    'Color', 'k');

h.Label = '\epsilon_{ss}';
h.FontSize = 18;

hold off
xlabel('\gamma t')
ylabel('\epsilon(\tau)')
xlim([0, max(gamma*t_list)])
ylim([0, 7])
set(gca, 'FontSize', 14)
grid on