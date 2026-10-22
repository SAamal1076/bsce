function info = quadriga_make_H(quadriga_src, out_prefix, array, K, varargin)
%QUADRIGA_MAKE_H  Raw 3GPP TR 38.901 channels (QuaDRiGa) for bsce, saved in parts as H (N x K complex).
%
%   quadriga_make_H(quadriga_src, out_prefix, array, K)
%   quadriga_make_H(quadriga_src, out_prefix, array, K, 'name', value, ...)
%
%   quadriga_src : the folder 'quadriga_src' of QuaDRiGa v2.8.1 (https://github.com/fraunhoferhhi/QuaDRiGa,
%                  compatible with 3GPP TR 38.901 v16.1). Runs in MATLAB and in Octave 8.
%   out_prefix   : files <out_prefix>_part001.mat, _part002.mat, ... ('part_users' channels each)
%   array        : 'UPA'    16 x 16 on the y, z axes, boresight +x  (= the DeepMIMO UPA of GM-LAMP / bsce)
%                  'ULA'    256 elements on the z axis (vertical)   (= GM-LAMP's DeepMIMO ULA)
%                  'ULA_Y'  256 elements on the y axis (horizontal) (= the usual 3GPP ULA; azimuth beams like SV)
%                  half-wavelength spacing in every case
%   K            : number of channels, e.g. 100000
%
%   Options (defaults = the bsce 3GPP setting of docs/3GPP_DATASET.md):
%     'scenario'   '3GPP_38.901_UMi'  UMi street canyon; LOS or NLOS drawn per user with the 38.901 LOS
%                                     probability (Table 7.4.2-1). Also '3GPP_38.901_UMi_LOS' (LOS only),
%                                     '3GPP_38.901_UMi_NLOS', '3GPP_38.901_UMa' (then set 'isd' 500, 'h_bs' 25)
%     'fc'         28e9               carrier [Hz] (= DeepMIMO O1_28 and the 28 GHz of the SV papers)
%     'element'    '3gpp'             BS element of 38.901 Table 7.3-1 (65 deg beamwidth, 8 dBi, 30 dB
%                                     front-back), vertical polarisation; 'omni' = isotropic elements
%     'isd'        200                inter-site distance [m]; users uniform in the 120-degree hexagonal
%                                     sector cell of that site (farthest user 2*isd/3)
%     'd_min'      10                 minimum 2-D BS-UE distance [m] (38.901 Table 7.8-1)
%     'h_bs'       10                 BS height [m] (UMi)
%     'h_ut'       1.5                UE height [m] (outdoor UE)
%     'indoor'     0                  fraction of outdoor-to-indoor users (the 38.901 system calibration
%                                     uses 0.8; DeepMIMO O1 has outdoor users only)
%     'drop_users' 500                users per drop: every drop is a new layout with new LSP maps
%     'part_users' 5000               users per output file (a multiple of drop_users)
%     'seed'       2026               drop d uses the seed seed + d, so any part can be made alone
%     'parts'      []                 make only these part numbers (e.g. 1:10 in one MATLAB, 11:20 in another)
%     'realized'   false              also compute the spreads of the generated paths (lsp_real; QuaDRiGa's
%                                     gen_lsf_from_ssf, about 50 ms per channel more; for the statistics check)
%     'table'      'v16'              UMi large-scale parameters: 'v16' = 38.901 v16.1, the tables QuaDRiGa
%                                     ships and is calibrated with; 'v19' = the Rel-19 update of DS, ASD, ASA,
%                                     ZSA (38.901 v19, valid 0.5-100 GHz; values as in Sionna 2.2 'v19_2')
%     'zsd_fix'    true               UMi LOS: add the term 0.01|h_UT - h_BS| of the 38.901 ZSD law
%                                     (Table 7.5-6) that QuaDRiGa's UMi LOS configuration leaves out
%     'czsd_fix'   true               UMi: intra-cluster zenith spread of departure (3/8) 10^mu_lgZSD per user
%                                     (eq. 7.5-20) instead of QuaDRiGa's fixed 4 deg
%     'zod_offset' true               UMi NLOS: the ZOD offset of Table 7.5-6, which QuaDRiGa does not apply
%   With the three fixes the elevation at the BS follows 38.901 (checked against Sionna's UMi in
%   docs/3GPP_DATASET.md). QuaDRiGa still draws the cluster angles its own way (it scales random
%   angles to the user's drawn spread, instead of eq. 7.5-9 to 7.5-19), so its azimuth spreads are
%   wider than those of the TR's procedure; scripts/sionna_make_H.py follows the TR as written.
%   Existing part files are kept (the run resumes) if they were made with the same settings; delete
%   them, or use another out_prefix, to change a setting.
%
%   Narrowband, one frequency (the carrier): h = sum over the paths of the channel coefficients,
%   as the single subcarrier of deepmimo_make_H.m. Plane waves, no mobility (use_3GPP_baseline = 1).
%   UE: one isotropic antenna. Every channel keeps its path loss and shadowing here; bsce normalises
%   ||x||^2 = N per channel (as for DeepMIMO), so the SNR is the per-channel SNR of SV and DeepMIMO.
%
%   Saved per part: H (N x Kp), loc (Kp x 3, BS at [0 0 h_bs]), los (Kp x 1), indoor (Kp x 1),
%   drop (Kp x 1), lsp (Kp x 7: DS [s], ASD, ASA, ZSD, ZSA [deg], K-factor [dB], SF [dB]: the
%   parameters QuaDRiGa drew from the 38.901 tables), lsp_real (Kp x 5: DS, ASD, ASA, ZSD, ZSA of the
%   generated paths; NaN unless 'realized' is true), lsp_names, params. Then in bsce:
%     python scripts/make_dataset.py --dataset 3GPP --array UPA --input "H_3gpp_upa_part*.mat" --n 100000
%
%   Example (MATLAB):
%     addpath('D:\...\bsce\matlab')
%     q = 'D:\...\QuaDRiGa\quadriga_src';
%     quadriga_make_H(q, 'D:\...\bsce\raw3gpp\H_3gpp_upa', 'UPA', 100000);
%     quadriga_make_H(q, 'D:\...\bsce\raw3gpp\H_3gpp_ulay', 'ULA_Y', 100000);

opt = struct('scenario', '3GPP_38.901_UMi', 'fc', 28e9, 'element', '3gpp', 'isd', 200, 'd_min', 10, ...
             'h_bs', 10, 'h_ut', 1.5, 'indoor', 0, 'drop_users', 500, 'part_users', 5000, 'seed', 2026, ...
             'parts', [], 'realized', false, 'table', 'v16', 'zsd_fix', true, 'czsd_fix', true, 'zod_offset', true);
if mod(numel(varargin), 2) ~= 0, error('options come in name, value pairs'); end
for i = 1:2:numel(varargin)
    key = lower(varargin{i});
    if ~isfield(opt, key), error('unknown option ''%s''', varargin{i}); end
    opt.(key) = varargin{i + 1};
end
if mod(opt.part_users, opt.drop_users) ~= 0, error('part_users must be a multiple of drop_users'); end
opt.table = lower(opt.table);
if ~any(strcmp(opt.table, {'v16', 'v19'})), error('table must be v16 or v19'); end
if strcmp(opt.table, 'v19') && isempty(strfind(upper(opt.scenario), 'UMI'))
    error('table v19 is implemented for the UMi scenarios only');
end
if strcmp(opt.table, 'v19') && opt.indoor > 0
    warning('table v19: the O2I users keep the v16.1 O2I parameters');
end

if ~is_absolute(quadriga_src), quadriga_src = fullfile(pwd, quadriga_src); end
if ~is_absolute(out_prefix), out_prefix = fullfile(pwd, out_prefix); end
if ~exist(fullfile(quadriga_src, '@qd_layout'), 'dir')
    error('quadriga_make_H:src', '%s is not the quadriga_src folder of QuaDRiGa (no @qd_layout inside)', quadriga_src);
end
addpath(quadriga_src);
out_dir = fileparts(out_prefix);
if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end

switch upper(array)
    case 'UPA',   Mv = 16;  Nh = 16;  g = [1 16 16];
    case 'ULA',   Mv = 256; Nh = 1;   g = [1 1 256];
    case 'ULA_Y', Mv = 1;   Nh = 256; g = [1 256 1];
    otherwise, error('array must be UPA, ULA or ULA_Y');
end
N = Mv * Nh;

s = qd_simulation_parameters;
s.center_frequency = opt.fc;
s.use_3GPP_baseline = 1;
s.show_progress_bars = 0;
tx = qd_arrayant('3gpp-3d', Mv, Nh, opt.fc, 1, 0, 0.5);   % element index: vertical (z) runs fastest
switch lower(opt.element)
    case '3gpp'
    case 'omni', tx.Fa(:) = 1; tx.Fb(:) = 0;
    otherwise, error('element must be 3gpp or omni');
end
rx = qd_arrayant('omni');

n_parts = ceil(K / opt.part_users);
parts = opt.parts;
if isempty(parts), parts = 1:n_parts; end
lsp_names = {'DS_s', 'ASD_deg', 'ASA_deg', 'ZSD_deg', 'ZSA_deg', 'KF_dB', 'SF_dB'};
params = struct('generator', 'bsce matlab/quadriga_make_H.m', ...
                'array', upper(array), 'num_ant_x', g(1), 'num_ant_y', g(2), 'num_ant_z', g(3), ...
                'ant_spacing', 0.5, 'scenario', opt.scenario, 'fc', opt.fc, 'element', lower(opt.element), ...
                'isd', opt.isd, 'd_min', opt.d_min, 'h_bs', opt.h_bs, 'h_ut', opt.h_ut, 'indoor_frac', opt.indoor, ...
                'drop_users', opt.drop_users, 'part_users', opt.part_users, 'seed', opt.seed, 'K_total', K, ...
                'narrowband', 1, 'use_3GPP_baseline', 1, 'ue_antenna', 'omni', 'realized', double(opt.realized), ...
                'table', opt.table, 'zsd_fix', double(opt.zsd_fix), 'czsd_fix', double(opt.czsd_fix), ...
                'zod_offset', double(opt.zod_offset), 'implementation', ...
                ['QuaDRiGa v2.8.1 (3GPP TR 38.901 ' opt.table ' tables)']);
same = {'array', 'scenario', 'fc', 'element', 'isd', 'd_min', 'h_bs', 'h_ut', 'indoor_frac', 'drop_users', ...
        'part_users', 'seed', 'table', 'zsd_fix', 'czsd_fix', 'zod_offset'};
fprintf(['3GPP %s (tables %s, fixes: ZSD %d, cluster ZSD %d, ZOD offset %d), %s, %.1f GHz, %s elements: ' ...
         '%d channels in %d part(s) of %d (drops of %d users)\n'], opt.scenario, opt.table, opt.zsd_fix, ...
        opt.czsd_fix, opt.zod_offset, upper(array), opt.fc / 1e9, lower(opt.element), K, n_parts, ...
        opt.part_users, opt.drop_users);
t_all = tic; made = 0;
for p = parts(:).'
    f = sprintf('%s_part%03d.mat', out_prefix, p);
    if exist(f, 'file')
        old = load(f, 'params');
        for i = 1:numel(same)
            b = params.(same{i});
            if isfield(old.params, same{i}), a = old.params.(same{i}); else, a = '(not stored)'; end
            if ~isequal(a, b)
                error('quadriga_make_H:settings', ['%s was made with %s = %s, this run has %s. Delete the old ' ...
                      'parts or use another out_prefix.'], f, same{i}, num2str(a), num2str(b));
            end
        end
        fprintf('part %d: %s exists, kept\n', p, f); continue;
    end
    first = (p - 1) * opt.part_users;                 % users before this part
    Kp = min(opt.part_users, K - first);
    if Kp <= 0, continue; end
    H = zeros(N, Kp); loc = zeros(Kp, 3); los = false(Kp, 1); indoor = false(Kp, 1); drop = zeros(Kp, 1);
    lsp = nan(Kp, 7); lsp_real = nan(Kp, 5);
    t_part = tic;
    for k0 = 0:opt.drop_users:Kp - 1
        n = min(opt.drop_users, Kp - k0);
        d = (first + k0) / opt.drop_users + 1;        % global drop number
        set_seed(opt.seed + d);
        [Hd, Ld, losd, ind, lspd, lrd] = one_drop(s, tx, rx, n, opt);
        ii = k0 + (1:n);
        H(:, ii) = Hd; loc(ii, :) = Ld; los(ii) = losd; indoor(ii) = ind; drop(ii) = d;
        lsp(ii, :) = lspd; lsp_real(ii, :) = lrd;
    end
    save(f, 'H', 'loc', 'los', 'indoor', 'drop', 'lsp', 'lsp_real', 'lsp_names', 'params', '-v7');
    made = made + Kp;
    el = toc(t_all);
    fprintf('part %d / %d: %d channels, LOS %.2f, %.0f s (%.1f ms per channel) -> %s\n', p, n_parts, Kp, ...
            mean(los), toc(t_part), 1000 * toc(t_part) / Kp, f);
    left = sum(parts > p) * opt.part_users;
    if left > 0, fprintf('          about %.0f min left\n', el / made * left / 60); end
end
info = params;
end

function [H, loc, los, indoor, lsp, lsp_real] = one_drop(s, tx, rx, n, opt)
% one drop: n users uniform in the hexagonal 120-degree sector cell, d2D >= d_min
r = opt.isd / 3;                                    % side of the sector hexagon; the BS is one of its corners
xy = zeros(2, 0);
while size(xy, 2) < n
    m = 4 * n;
    c = [2 * r * rand(1, m); (2 * rand(1, m) - 1) * r * sqrt(3) / 2];
    q = abs(c - [r; 0]);
    in = q(2, :) <= r * sqrt(3) / 2 & sqrt(3) * q(1, :) + q(2, :) <= sqrt(3) * r & sum(c .^ 2, 1) >= opt.d_min ^ 2;
    xy = [xy, c(:, in)]; %#ok<AGROW>
end
xy = xy(:, 1:n);
l = qd_layout(s);
l.tx_position = [0; 0; opt.h_bs];
l.tx_array = tx;
l.rx_array = rx;
l.no_rx = n;
l.rx_position = [xy; opt.h_ut * ones(1, n)];
indoor = l.set_scenario(opt.scenario, [], [], opt.indoor);
indoor = logical(indoor(:));
if numel(indoor) ~= n, indoor = false(n, 1); end
b = l.init_builder;
adjust_tables(b, opt);
gen_parameters(b);
if opt.realized
    br = copy(b);
    gen_lsf_from_ssf(br);                           % spreads of the generated paths
end
groups = elevation_fixes(b, opt);                   % builder groups (one builder per user with 'czsd_fix')
H = zeros(tx.no_elements, n); los = false(n, 1); lsp = nan(n, 7); lsp_real = nan(n, 5);
for g = 1:numel(groups)
    c = get_channels(groups{g});
    for i = 1:numel(c)
        k = rx_index(c(i).name);
        H(:, k) = reshape(sum(c(i).coeff(1, :, :, 1), 3), [], 1);
    end
end
for j = 1:numel(b)
    is_los = isempty(strfind(upper(b(j).name), 'NLOS'));
    for i = 1:b(j).no_rx_positions
        k = rx_index(b(j).rx_track(1, i).name);
        los(k) = is_los;
        lsp(k, :) = [b(j).ds(i), b(j).asD(i), b(j).asA(i), b(j).esD(i), b(j).esA(i), ...
                     10 * log10(max(b(j).kf(i), 1e-10)), 10 * log10(b(j).sf(i))];
        if opt.realized
            lsp_real(k, :) = [br(j).ds(i), br(j).asD(i), br(j).asA(i), br(j).esD(i), br(j).esA(i)];
        end
    end
end
loc = l.rx_position.';
end

function adjust_tables(b, opt)
% 38.901 corrections of the UMi tables (see the options 'zsd_fix' and 'table')
v19 = struct('LOS', [-0.18 -7.28 0 0.39; -0.05 1.21 0.08 0.29; -0.07 1.66 0.021 0.26; -0.11 0.81 -0.03 0.29], ...
             'NLOS', [-0.22 -6.87 0.19 0.22; -0.24 1.54 0.10 0.33; -0.07 1.76 0.05 0.27; -0.03 0.92 -0.05 0.35]);
f = {'DS', 'AS_D', 'AS_A', 'ES_A'};      % rows: mu = a log10(1 + fc) + b, sigma = c log10(1 + fc) + d
for j = 1:numel(b)
    nm = upper(b(j).name);
    if isempty(strfind(nm, 'UMI')), continue; end
    is_los = isempty(strfind(nm, 'NLOS'));
    sp = b(j).scenpar;
    if opt.zsd_fix && is_los              % lgZSD = max(-0.21, -14.8 d2D/1000 + 0.01|hUT - hBS| + 0.83)
        sp.ES_D_mu = sp.ES_D_mu + 0.01 * abs(opt.h_ut - opt.h_bs);
    end
    if strcmp(opt.table, 'v19') && isempty(strfind(nm, 'O2I'))
        if is_los, v = v19.LOS; else, v = v19.NLOS; end
        for i = 1:4
            sp.([f{i} '_gamma']) = v(i, 1); sp.([f{i} '_mu']) = v(i, 2);
            sp.([f{i} '_delta']) = v(i, 3); sp.([f{i} '_sigma']) = v(i, 4); sp.([f{i} '_omega']) = 1;
        end
    end
    b(j).scenpar = sp;
end
end

function groups = elevation_fixes(b, opt)
% 38.901 elevation of departure for UMi (see the options 'zod_offset' and 'czsd_fix'); b after gen_parameters
groups = {};
for j = 1:numel(b)
    nm = upper(b(j).name);
    umi = ~isempty(strfind(nm, 'UMI'));
    if ~umi || (~opt.zod_offset && ~opt.czsd_fix), groups{end + 1} = b(j); continue; end %#ok<AGROW>
    is_los = isempty(strfind(nm, 'NLOS'));
    p = b(j).rx_positions;
    d2 = sqrt(sum(p(1:2, :) .^ 2, 1));
    if opt.zod_offset && ~is_los
        % Table 7.5-6, NLOS: mu_offset,ZOD = -10^(-1.5 log10(max(10, d2D)) + 3.3) deg, added to the cluster zeniths
        % (eq. 7.5-19); QuaDRiGa uses elevation = 90 deg - zenith. Column 1 is QuaDRiGa's (zero-power) LOS slot.
        off = -10 .^ (-1.5 * log10(max(10, d2)) + 3.3);
        E = b(j).EoD;
        E(:, 2:end) = E(:, 2:end) - (off(:) * pi / 180) * ones(1, size(E, 2) - 1);
        b(j).EoD = E;
    end
    if opt.czsd_fix
        % eq. 7.5-20: ray zeniths of departure = cluster zenith + (3/8) 10^mu_lgZSD alpha_m, mu_lgZSD of the user's
        % distance (QuaDRiGa uses a fixed PerClusterES_D of 4 deg for UMi)
        sp = b(j).scenpar;
        mu = max(sp.ES_D_mu_min, sp.ES_D_mu + sp.ES_D_mu_A * d2 / 1000);
        bs = split_rx(b(j));
        for i = 1:numel(bs)
            q = bs(i).scenpar;
            q.PerClusterES_D = 3 / 8 * 10 ^ mu(i);
            bs(i).scenpar_nocheck = q;
        end
        groups{end + 1} = bs; %#ok<AGROW>
    else
        groups{end + 1} = b(j); %#ok<AGROW>
    end
end
end

function k = rx_index(name)
t = regexp(name, 'Rx(\d+)$', 'tokens', 'once');
if isempty(t), error('cannot read the receiver index from ''%s''', name); end
k = str2double(t{1});
end

function set_seed(v)
if exist('OCTAVE_VERSION', 'builtin')
    rand('state', v); randn('state', v); %#ok<RAND>
else
    rng(v, 'twister');
end
end

function t = is_absolute(p)
t = ~isempty(p) && (p(1) == '/' || p(1) == '\' || (numel(p) > 1 && p(2) == ':'));
end
