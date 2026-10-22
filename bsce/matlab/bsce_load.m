function net = bsce_load(file)
% BSCE_LOAD  Load a model exported by bsce (python): net = bsce_load('model.mat')
% Use with bsce_infer(net, y). Works in MATLAB and GNU Octave.
net = load(file);
if isfield(net, 'model') && ~ischar(net.model), net.model = char(net.model); end
net.model = strtrim(net.model);
if isfield(net, 'operator'), net.operator = strtrim(char(net.operator)); end
net.T = double(net.T);
end
