function quadriga_strict_test(quadriga_src, arrname, out, nusers, czsd_fix, zod_fix)
addpath(quadriga_src);
s = qd_simulation_parameters; s.center_frequency = 28e9; s.use_3GPP_baseline = 1; s.show_progress_bars = 0;
switch arrname
  case 'UPA', tx = qd_arrayant('3gpp-3d', 16, 16, 28e9, 1, 0, 0.5);
  case 'ULA', tx = qd_arrayant('3gpp-3d', 256, 1, 28e9, 1, 0, 0.5);
  case 'ULA_Y', tx = qd_arrayant('3gpp-3d', 1, 256, 28e9, 1, 0, 0.5);
end
rx = qd_arrayant('omni');
H = []; LOS = []; LOC = [];
for d = 1:ceil(nusers/250)
  rand('state', 100+d); randn('state', 100+d);
  n = 250; r = 200/3; xy = zeros(2,0);
  while size(xy,2) < n
    c = [2*r*rand(1,4*n); (2*rand(1,4*n)-1)*r*sqrt(3)/2]; q = abs(c - [r;0]);
    in = q(2,:) <= r*sqrt(3)/2 & sqrt(3)*q(1,:)+q(2,:) <= sqrt(3)*r & sum(c.^2,1) >= 100; xy = [xy c(:,in)];
  end
  xy = xy(:,1:n);
  l = qd_layout(s); l.tx_position = [0;0;10]; l.tx_array = tx; l.rx_array = rx; l.no_rx = n; l.rx_position = [xy; 1.5*ones(1,n)];
  l.set_scenario('3GPP_38.901_UMi', [], [], 0);
  b = l.init_builder;
  for j = 1:numel(b)
    sp = b(j).scenpar; if isempty(strfind(upper(b(j).name),'NLOS')), sp.ES_D_mu = sp.ES_D_mu + 0.085; end; b(j).scenpar = sp;
  end
  gen_parameters(b);
  bb = {};
  for j = 1:numel(b)
    is_los = isempty(strfind(upper(b(j).name),'NLOS'));
    pos = b(j).rx_positions; d2 = sqrt(sum(pos(1:2,:).^2,1));
    if zod_fix && ~is_los
      off = -10.^(-1.5*log10(max(10,d2)) + 3.3);          % zenith offset [deg]; elevation = -zenith
      E = b(j).EoD; E(:,2:end) = E(:,2:end) - (off(:)*pi/180)*ones(1,size(E,2)-1); b(j).EoD = E;
    end
    if czsd_fix
      sp = b(j).scenpar;
      mu = max(sp.ES_D_mu_min, sp.ES_D_mu + sp.ES_D_mu_A*d2/1000);
      bs = split_rx(b(j));
      for i = 1:numel(bs), p = bs(i).scenpar; p.PerClusterES_D = 3/8*10^mu(i); bs(i).scenpar_nocheck = p; end
      bb{end+1} = bs;
    else
      bb{end+1} = b(j);
    end
  end
  Hd = zeros(tx.no_elements, n); losd = false(1,n); t = 0;
  for g = 1:numel(bb)
    tic; c = get_channels(bb{g}); t = t + toc;
    for i = 1:numel(c)
      tk = regexp(c(i).name,'Rx(\d+)$','tokens','once'); k = str2double(tk{1});
      Hd(:,k) = reshape(sum(c(i).coeff(1,:,:,1),3),[],1);
      losd(k) = isempty(strfind(upper(c(i).name),'NLOS'));
    end
  end
  fprintf('drop %d: %.1f s for channels (%d builder groups)\n', d, t, numel(bb));
  H = [H Hd]; LOS = [LOS losd]; LOC = [LOC; l.rx_position.'];
end
save(out, 'H', 'LOS', 'LOC', '-v7');
end
