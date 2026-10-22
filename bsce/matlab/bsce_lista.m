function [xhat, layers] = bsce_lista(net, y)
% LISTA inference: x_t = shrink(W_t x_{t-1} + B y, theta_t) (ALISTA-repo LISTA).
T = net.T; B = net.B; N = size(B, 1);
untied = ~isfield(net, 'W');
cm = ~isfield(net, 'cfg_complex_mode') || double(net.cfg_complex_mode) ~= 0;
By = B*y; xh = zeros(N, size(y,2)); layers = cell(1, T);
for t = 1:T
  if untied, W = net.(sprintf('W_%d', t)); else, W = net.W; end
  th = net.(sprintf('theta_%d', t));
  xh = bsce_soft(W*xh + By, th, cm);
  layers{t} = xh;
end
xhat = xh;
end
