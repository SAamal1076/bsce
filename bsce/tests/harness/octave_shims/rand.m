function r = rand(varargin)
d = bsce_dims(varargin{:});
r = reshape(bsce_stream('u', prod(d)), d);
end
