function [xhat, layers] = bsce_lamp(net, y)
% LAMP / GM-LAMP inference (same equations as LAMP.m / GM_LAMP.m of Wei-Hu-Dai, any T).
A = double(net.A); [M, N] = size(A); T = net.T;
gm = strcmp(net.model, 'GM-LAMP');
layers = cell(1, T);
v = y;
B = net.('B_0'); theta = net.('theta0');
rvar = (1/M)*sum(abs(y).^2, 1);
[xhat, b, c] = shrink(B*y, rvar, theta, gm, N, M);
layers{1} = xhat;
for t = 1:T-1
  B = net.(sprintf('B_%d', t)); theta = net.(sprintf('theta%d', t));
  v = y - A*xhat + bsxfun(@times, v, b) + bsxfun(@times, conj(v), c);
  rhat = xhat + B*v;
  rvar = 1/M*sum(abs(v).^2, 1);
  [xhat, b, c] = shrink(rhat, rvar, theta, gm, N, M);
  layers{t+1} = xhat;
end
end

function [xhat, b, c] = shrink(r, rvar, theta, gm, N, M)
if ~gm
  lam = theta(1)*sqrt(rvar); lam = max(abs(lam), 0);
  xhat = (r./abs(r)).*max(bsxfun(@minus, abs(r), lam), 0);
  on = bsxfun(@gt, abs(r), lam);
  b = (1/M)*sum(on.*(1 - bsxfun(@rdivide, lam, 2*abs(r)+eps)), 1);
  c = (1/M)*sum(on.*(0.5*bsxfun(@times, lam, sqrt(r)).*(1./((conj(r)).^(3/2)+eps))), 1);
else
  [xhat, dxdr, dxdr1] = eta_gm_bsce(r, rvar, theta);
  b = (N/M)*dxdr; c = (N/M)*dxdr1;
end
end

function [xhat, dxdr, dxdr1] = eta_gm_bsce(rhat, rvar, theta)
% identical to eta_gm.m (K = number of columns of theta)
th = -40; K = size(theta, 2);
P = exp(theta(1,:))./sum(exp(theta(1,:)));
mu = theta(2,:);
var = exp(max(th, theta(3,:)));
U = 0; V = 0; dU = 0; dV = 0; dU1 = 0; dV1 = 0;
for k = 1:K
  tilde_mu_k_x = (mu(k)*rvar + rhat*var(k))./(rvar + var(k));
  CNpdf_k = exp(max(th, real(-conj(rhat-mu(k)).*(rhat-mu(k))./2./(rvar+var(k)))))./2./pi./(var(k)+rvar);
  u_k = tilde_mu_k_x.*P(k).*CNpdf_k;
  du_k = var(k)./(rvar+var(k)).*P(k).*CNpdf_k + u_k.*(-conj(rhat-mu(k))./2./(rvar+var(k)));
  du_k1 = u_k.*(-(rhat-mu(k))./2./(rvar+var(k)));
  v_k = P(k).*CNpdf_k;
  dv_k = v_k.*(-conj(rhat-mu(k))./2./(rvar+var(k)));
  dv_k1 = v_k.*(-(rhat-mu(k))./2./(rvar+var(k)));
  U = U + u_k; dU = dU + du_k; dU1 = dU1 + du_k1;
  V = V + v_k; dV = dV + dv_k; dV1 = dV1 + dv_k1;
end
xhat = U./V;
dxdr = mean((dU.*V - U.*dV)./(V.*conj(V)), 1);
dxdr1 = mean((dU1.*V - U.*dV1)./(V.*conj(V)), 1);
end
