function [xhat, layers] = bsce_lvamp(net, y)
% LVAMP (SVD parameterisation) inference with the Bernoulli-Gaussian denoiser.
V = net.V; rS2 = net.rS2; UsT = net.UsT; T = net.T; N = size(V, 1);
cm = ~isfield(net, 'cfg_complex_mode') || double(net.cfg_complex_mode) ~= 0;
rj = zeros(N, size(y,2));
taurj = sum(abs(y).^2, 1)/N;
yvar = exp(double(net.logyvar));
ytilde = UsT*y; layers = cell(1, T);
for t = 0:T-1
  varRat = yvar./taurj;
  scale_each = 1./(1 + rS2*varRat);
  zetai = N./sum(scale_each, 1);
  adjust = bsxfun(@times, scale_each.*(ytilde - V.'*rj), zetai);
  ri = rj + V*adjust;
  tauri = taurj.*(zetai - 1);
  th = net.(sprintf('theta_%d', t));
  [xh, dxdr] = bg(ri, tauri, th, cm);
  layers{t+1} = xh;
  zetaj = 1./(1 - dxdr);
  rj = bsxfun(@times, xh - bsxfun(@times, dxdr, ri), zetaj);
  taurj = tauri.*(zetaj - 1);
end
xhat = xh;
end

function [xhat, dxdr] = bg(r, rvar, theta, cm)
xvar1 = abs(theta(1)); loglam = theta(2);
beta = 1./(1 + rvar/xvar1);
if ~cm
  r2scale = bsxfun(@times, r.*r, beta./rvar);
  rho = exp(loglam - .5*r2scale).*repmat(sqrt(1 + xvar1./rvar), size(r,1), 1);
  rho1 = rho + 1;
  xhat = bsxfun(@times, beta, r)./rho1;
  dxdr = mean(bsxfun(@times, beta, (1 + rho.*(1 + r2scale))./(rho1.^2)), 1);
else
  r2scale = bsxfun(@times, real(r.*conj(r)), beta./rvar);
  rho = exp(loglam - r2scale).*repmat(1 + xvar1./rvar, size(r,1), 1);
  rho1 = rho + 1;
  g = bsxfun(@rdivide, beta, rho1);
  xhat = g.*r;
  dxdr = mean(g.*(1 + rho.*r2scale./rho1), 1);
end
end
