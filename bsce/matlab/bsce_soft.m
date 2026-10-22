function z = bsce_soft(r, lam, complex_mode)
% soft threshold with lam = max(lam,0); complex: r/|r|*max(|r|-lam,0) (phase kept)
lam = max(lam, 0);
a = abs(r);
ampl = max(bsxfun(@minus, a, lam), 0);
if complex_mode
  sc = zeros(size(a)); nz = a > 0; sc(nz) = ampl(nz)./a(nz);
  z = r.*sc;
else
  z = sign(r).*ampl;
end
end
