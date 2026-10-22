function [xhat, layers] = bsce_ladmm(net, y)
% LADMM inference (same equations as LADMM_test.m; structured operators are exported
% materialised as H_matrix_t, so every operator runs through this function). The threshold options other
% than 'scalar', data_step 'whitened', noise_level and debias run in Python only: refused here, never approximated.
opt = @(f, d) isfield(net, f) && ~strcmp(strtrim(char(net.(f))), d);
if opt('cfg_threshold', 'scalar') || opt('cfg_data_step', 'plain') || opt('cfg_noise_level', 'median') ...
    || opt('cfg_debias', 'none')
  error('bsce_ladmm: this LADMM (threshold / data_step / noise_level / debias option) runs in Python only');
end
A = double(net.A); [M, N] = size(A); T = net.T;
Z = zeros(N, size(y,2)); U = Z; AHy = A'*y;
layers = cell(1, T);
for t = 0:T-1
  H_matrix = net.(sprintf('H_matrix_%d', t)); rho = double(net.(sprintf('rho_%d', t)));
  theta = net.(sprintf('theta_%d', t)); alpha = double(net.(sprintf('alpha_%d', t)));
  H = H_matrix*(AHy + rho*Z - U);
  V = H + U/rho;
  thr = theta(1)/rho;
  mag = abs(V); ph = angle(V);
  sh = max(mag - thr, 0);
  Z = sh.*exp(1j*ph); Z(sh == 0) = 0;
  U = U + alpha*rho*(H - Z);
  layers{t+1} = Z;
end
xhat = Z;
end
