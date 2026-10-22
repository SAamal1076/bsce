function r = randn(varargin)
d = bsce_dims(varargin{:});
r = reshape(bsce_stream('n', prod(d)), d);
end
