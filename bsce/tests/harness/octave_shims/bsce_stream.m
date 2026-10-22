function out = bsce_stream(kind, count)
% Replay pre-generated random numbers (used to test original .m files unchanged).
% The stream file is given by environment variable BSCE_STREAM (.mat with u, n).
persistent U NN iu in_
if isempty(U)
  S = load(getenv('BSCE_STREAM'));
  U = S.u(:); NN = S.n(:); iu = 0; in_ = 0;
end
if strcmp(kind, 'u')
  if iu + count > numel(U), error('bsce_stream: uniform stream exhausted'); end
  out = U(iu+1:iu+count); iu = iu + count;
else
  if in_ + count > numel(NN), error('bsce_stream: normal stream exhausted'); end
  out = NN(in_+1:in_+count); in_ = in_ + count;
end
end
