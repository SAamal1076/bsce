function [xhat, layers] = bsce_infer(net, y)
% BSCE_INFER  Run a bsce-trained network on measurements y (M x K complex).
%   [xhat, layers] = bsce_infer(net, y), net = bsce_load('file.mat').
%   layers{t} is the estimate after layer t.
y = double(y);
switch net.model
  case {'LAMP', 'GM-LAMP'}, [xhat, layers] = bsce_lamp(net, y);
  case 'LADMM',             [xhat, layers] = bsce_ladmm(net, y);
  case 'LISTA',             [xhat, layers] = bsce_lista(net, y);
  case 'ALISTA',            [xhat, layers] = bsce_alista(net, y);
  case 'LVAMP',             [xhat, layers] = bsce_lvamp(net, y);
  case 'ADMM-Net',          [xhat, layers] = bsce_admmnet(net, y);
  otherwise, error('bsce_infer: unknown model %s', net.model);
end
end
