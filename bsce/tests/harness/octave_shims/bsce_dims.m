function d = bsce_dims(varargin)
if numel(varargin) == 0, d = [1 1];
elseif numel(varargin) == 1
  v = varargin{1}; if isscalar(v), d = [v v]; else, d = v(:)'; end
else, d = cell2mat(varargin);
end
end
