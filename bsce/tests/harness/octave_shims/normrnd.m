function r = normrnd(mu, sigma, varargin)
d = bsce_dims(varargin{:});
r = mu + sigma .* reshape(bsce_stream('n', prod(d)), d);
end
