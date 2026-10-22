function H = deepmimo_make_H(deepmimo_dir, out_file, array, rows, num_paths, bs, scenario)
%DEEPMIMO_MAKE_H  Raw DeepMIMO v1.3 channels for bsce (GM-LAMP setting), saved as H (N x K complex).
%
%   deepmimo_make_H(deepmimo_dir, out_file, array)
%   deepmimo_make_H(deepmimo_dir, out_file, array, rows, num_paths, bs, scenario)
%
%   deepmimo_dir : folder of DeepMIMO-matlab-1.3 (contains DeepMIMO_Dataset_Generator.m,
%                  'MAT functions' and 'RayTracing Scenarios/<scenario>/...')
%   out_file     : .mat file to write (saved with -v7 so scipy can read it)
%   array        : 'ULA'   256 elements on the z axis (vertical) = GM-LAMP's DeepMIMO ULA
%                  'UPA'   1 x 16 x 16 on the y, z axes          = GM-LAMP's DeepMIMO UPA
%                  'ULA_Y' / 'ULA_X'  256 elements on the y / x axis (not GM-LAMP's setting)
%                  (axis identified by comparing the strongest-beam statistics with GM-LAMP's
%                  ULA_test_channel.mat: z axis TV 0.17 vs y axis 0.98; see deepmimo_check)
%   rows         : [first last] user rows, default [1000 1300]  (301 x 181 = 54,481 users)
%   num_paths    : default 3
%   bs           : active BS, default 3
%   scenario     : default 'O1_28'
%
%   Calls the unchanged DeepMIMO_generator.m of v1.3 (half-wave dipole pattern applied) with
%   ant_spacing 0.5 and a single subcarrier (OFDM_limit = 1 -> k = 0, narrowband).
%   Saved variables: H (N x K), loc (K x 3), params.  Then in bsce:
%     python scripts/deepmimo_convert.py --scenario configs/scenarios/deepmimo_ula.yaml --input H_ula.mat
%
%   Example (MATLAB, from any folder):
%     addpath('D:\...\bsce\matlab')
%     dm = 'D:\Academic Profile\LADMM-MIMO\codes\DeepMIMO-matlab-1.3';
%     deepmimo_make_H(dm, 'D:\...\bsce\H_ula.mat', 'ULA');
%     deepmimo_make_H(dm, 'D:\...\bsce\H_upa.mat', 'UPA');

if nargin < 4 || isempty(rows),      rows = [1000 1300]; end
if nargin < 5 || isempty(num_paths), num_paths = 3;      end
if nargin < 6 || isempty(bs),        bs = 3;             end
if nargin < 7 || isempty(scenario),  scenario = 'O1_28'; end

old = pwd;
if ~is_absolute(out_file), out_file = fullfile(old, out_file); end
if ~is_absolute(deepmimo_dir), deepmimo_dir = fullfile(old, deepmimo_dir); end
scen_dir = fullfile(deepmimo_dir, 'RayTracing Scenarios', scenario);
if ~exist(fullfile(scen_dir, [scenario '.params.mat']), 'file')
    error('deepmimo_make_H:scenario', ['%s.params.mat not found in\n  %s\nPut the extracted %s ' ...
          'folder there (the .mat files directly inside it).'], scenario, scen_dir, scenario);
end

params.scenario = scenario;
params.active_BS = bs;
params.active_user_first = rows(1);
params.active_user_last = rows(2);
switch upper(array)
    case 'ULA',   params.num_ant_x = 1;   params.num_ant_y = 1;   params.num_ant_z = 256;
    case 'ULA_Y', params.num_ant_x = 1;   params.num_ant_y = 256; params.num_ant_z = 1;
    case 'ULA_X', params.num_ant_x = 256; params.num_ant_y = 1;   params.num_ant_z = 1;
    case 'UPA', params.num_ant_x = 1; params.num_ant_y = 16;  params.num_ant_z = 16;
    otherwise, error('array must be ULA, UPA, ULA_Y or ULA_X');
end
params.ant_spacing = 0.5;
params.bandwidth = 0.5;
params.num_OFDM = 1024;
params.OFDM_sampling_factor = 1;
params.OFDM_limit = 1;
params.num_paths = num_paths;
params.saveDataset = 0;

addpath(fullfile(deepmimo_dir, 'MAT functions'));
cd(deepmimo_dir);                      % the generator uses paths relative to its own folder
try
    [ds, params] = DeepMIMO_generator(params);
catch err
    cd(old); rethrow(err);
end
cd(old);

U = ds{1}.user;
K = numel(U);
N = params.num_ant_x * params.num_ant_y * params.num_ant_z;
H = zeros(N, K);
loc = zeros(K, 3);
for k = 1:K
    H(:, k) = U{k}.channel(:, 1);
    loc(k, :) = U{k}.loc;
end
nz = sum(all(H == 0, 1));
fprintf('\n%s %s BS %d rows %d-%d, %d paths: K = %d users, %d with no path (zero channel)\n', ...
        scenario, upper(array), bs, rows(1), rows(2), num_paths, K, nz);
if isequal(rows, [1000 1300]) && K - nz < 54000
    warning('only %d non-zero channels; bsce needs 54000 (50000/2000/2000). Extend rows.', K - nz);
end
save(out_file, 'H', 'loc', 'params', '-v7');
fprintf('saved %s  (H: %d x %d complex)\n', out_file, N, K);
end

function t = is_absolute(p)
t = ~isempty(p) && (p(1) == '/' || p(1) == '\' || (numel(p) > 1 && p(2) == ':'));
end
