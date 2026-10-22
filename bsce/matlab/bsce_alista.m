function [xhat, layers] = bsce_alista(net, y)
% ALISTA inference: z = x + alpha_t W'(y - A x); x = shrink_ss(z, theta_t, p_t).
A = double(net.A); W = net.W; T = net.T; N = size(A, 2); ps = net.ps(:)';
cm = ~isfield(net, 'cfg_complex_mode') || double(net.cfg_complex_mode) ~= 0;
xh = zeros(N, size(y,2)); layers = cell(1, T);
for t = 1:T
  alpha = double(net.(sprintf('alpha_%d', t))); th = net.(sprintf('theta_%d', t));
  res = y - A*xh;
  zh = xh + alpha*(W.'*res);
  xh = shrink_ss(zh, th, ps(t), cm);
  layers{t} = xh;
end
xhat = xh;
end

function out = shrink_ss(in, theta, q, cm)
% support-selection shrinkage; percentile with TF 'nearest' (round half to even) rule
ab = abs(in); n = size(ab, 1);
pos = (n-1)*(1 - (100-q)/100);
idx = round(pos);
if abs(pos - fix(pos)) == 0.5 && mod(idx, 2) == 1, idx = idx - sign(pos); end
srt = sort(ab, 1, 'descend');
thres = srt(idx+1, :);
index = double(bsxfun(@gt, ab, theta) & bsxfun(@gt, ab, thres));
shr_in = (1-index).*in;
out = index.*in + bsce_soft(shr_in, theta, cm);
end
