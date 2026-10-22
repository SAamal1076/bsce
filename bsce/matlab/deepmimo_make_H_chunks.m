function files = deepmimo_make_H_chunks(deepmimo_dir, out_dir, array, n_users, chunk_rows, first_row)
%DEEPMIMO_MAKE_H_CHUNKS  Raw DeepMIMO channels for a LARGE data set (e.g. 200,000 users), in row chunks.
%
%   files = deepmimo_make_H_chunks(deepmimo_dir, out_dir, array)              % 200,000 users
%   files = deepmimo_make_H_chunks(deepmimo_dir, out_dir, array, n_users, chunk_rows, first_row)
%
%   deepmimo_dir : folder of DeepMIMO-matlab-1.3 (with 'RayTracing Scenarios\O1_28\O1_28.params.mat' ...)
%   out_dir      : folder for the files H_<array>_part<k>.mat (created if needed)
%   array        : 'ULA' (256 on z, GM-LAMP) or 'UPA' (16 x 16 on y-z, GM-LAMP)
%   n_users      : users wanted in train + val + test, default 200000 (+ the 2000 GM-LAMP test users are
%                  inside rows 1000-1300 and are removed from train/val by make_dataset.py)
%   chunk_rows   : rows per file, default 300 (about 54k users and ~110 MB per file; lower it if MATLAB runs out
%                  of memory)
%   first_row    : default 1000 (GM-LAMP's first row; keep 1000 so that GM-LAMP's test users are included)
%
%   O1_28, BS 3, 3 paths, one subcarrier, 181 users per row (rows 1..2751 of region R1): 200,000 users need
%   rows 1000-2105. Rows outside 1000-1300 lie outside GM-LAMP's user area: the larger set is a little
%   different from GM-LAMP's test users (make_dataset.py prints and plots the train-test difference).
%
%   Then in Python (bsce folder):
%     python scripts/make_dataset.py --dataset DeepMIMO --array UPA --input "<out_dir>/H_upa_part*.mat" --n 200000 --snr 0 20

if nargin < 4 || isempty(n_users),    n_users = 200000; end
if nargin < 5 || isempty(chunk_rows), chunk_rows = 300;  end
if nargin < 6 || isempty(first_row),  first_row = 1000;  end
per_row = 181;
need_rows = ceil((n_users + 2000) / per_row) + 5;        % + GM-LAMP test users + users without a path
last_row = first_row + need_rows - 1;
if last_row > 2751
    error('rows %d-%d needed, but region R1 of O1 ends at row 2751: lower n_users or first_row', first_row, last_row);
end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
starts = first_row:chunk_rows:last_row;
files = cell(1, numel(starts));
for k = 1:numel(starts)
    r = [starts(k), min(starts(k) + chunk_rows - 1, last_row)];
    files{k} = fullfile(out_dir, sprintf('H_%s_part%d.mat', lower(array), k));
    if exist(files{k}, 'file')
        fprintf('keep %s (delete it to regenerate)\n', files{k});
        continue
    end
    fprintf('\n=== part %d / %d: rows %d-%d ===\n', k, numel(starts), r(1), r(2));
    deepmimo_make_H(deepmimo_dir, files{k}, array, r);
end
fprintf('\n%d files in %s (rows %d-%d, about %d users)\n', numel(files), out_dir, first_row, last_row, ...
        (last_row - first_row + 1) * per_row);
end
