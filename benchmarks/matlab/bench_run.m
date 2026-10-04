function [out1, out2, times] = bench_run(fn, n_runs, n_warmup)
% BENCH_RUN  Time `fn()` and return best-of-N statistics.
%
%   [out1, out2, times] = bench_run(fn, n_runs, n_warmup)
%
% Inputs:
%   fn       - zero-argument function handle that returns the result
%   n_runs   - number of timed runs (default 5)
%   n_warmup - number of untimed warmup runs (default 1)
%
% Outputs:
%   out1, out2 - the first two return values of the LAST run of fn()
%                (use [] if fn returns less than 2 outputs)
%   times      - vector of n_runs timed wall-clock seconds

if nargin < 2, n_runs = 5; end
if nargin < 3, n_warmup = 1; end

% Warmup
for i = 1:n_warmup
  [out1, out2] = capture(fn);
end

% Timed runs
times = zeros(1, n_runs);
for i = 1:n_runs
  t0 = tic;
  [out1, out2] = capture(fn);
  times(i) = toc(t0);
end
end

function [a, b] = capture(fn)
% All benchmark callables must expose exactly two outputs.  Keeping this
% contract explicit prevents real simulator errors from being mistaken for
% output-arity errors and prevents one-output functions from running twice.
[a, b] = fn();
end
