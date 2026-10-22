function [xhat, layers] = bsce_admmnet(net, y)
% ADMM-Net (Klioui) inference; Winv_t = inv(W_t + max(rho_t,0) I) is exported by bsce.
A = double(net.A); T = net.T; N = size(A, 2);
yf = A'*y; x = zeros(N, size(y,2)); z = x; v = x; layers = cell(1, T);
for t = 0:T-1
  tt = max(double(net.(sprintf('rho_%d', t))), 0);
  beta = double(net.(sprintf('beta_%d', t)));
  x = net.(sprintf('Winv_%d', t))*(tt*(z - v) + yf);
  u = x + v;
  z = exp(1j*angle(u)).*max(abs(u) - beta, 0);
  v = v + x - z;
  layers{t+1} = x;
end
xhat = x;
end
