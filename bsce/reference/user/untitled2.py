# -*- coding: utf-8 -*-
"""
Train_LADMM.py - PyTorch version
Exact same algorithm as TF version, runs on GPU on Windows
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as la
from scipy.io import loadmat

# ✅ Global dtype — complex64/float32 fully GPU-supported in PyTorch
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLOAT   = torch.float32
COMPLEX = torch.complex64

print('Using device:', DEVICE)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')


# ──────────────────────────────────────────────
# Shrinkage
# ──────────────────────────────────────────────
def complex_soft_threshold(V, threshold):
    """Complex soft thresholding — identical math to TF version"""
    magnitude = torch.abs(V)                                    # complex64 → float32
    phase     = torch.angle(V)                                  # complex64 → float32

    shrunk_magnitude = torch.clamp(magnitude - threshold, min=0.0)

    exp_factor = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    Z = shrunk_magnitude.to(COMPLEX) * exp_factor

    return Z


# ──────────────────────────────────────────────
# LADMM Network
# ──────────────────────────────────────────────
class LADMM(nn.Module):
    def __init__(self, A, T, shrink='soft', untied=True):
        super(LADMM, self).__init__()
        M, N = A.shape
        self.M       = M
        self.N       = N
        self.T       = T
        self.shrink  = shrink
        self.untied  = untied

        # Register A as buffer (not trained, but moves to GPU with .to(device))
        A_f = torch.tensor(A, dtype=FLOAT)
        self.register_buffer('A',  A_f)
        self.register_buffer('AT', A_f.T.contiguous())

        # Pre-compute initial H matrix
        AHA              = A.T @ A
        initial_H_matrix = la.inv(AHA + np.eye(N)).astype(np.float32)
        init_H           = torch.tensor(initial_H_matrix, dtype=FLOAT)

        # Learnable parameters — same variables as TF version
        self.H_matrices = nn.ParameterList()
        self.rhos       = nn.ParameterList()
        self.thetas     = nn.ParameterList()
        self.alphas     = nn.ParameterList()

        for t in range(T):
            if untied or t == 0:
                self.H_matrices.append(nn.Parameter(init_H.clone()))
            self.rhos.append(  nn.Parameter(torch.ones(1,  dtype=FLOAT)))
            self.thetas.append(nn.Parameter(torch.ones(1,  dtype=FLOAT)))
            self.alphas.append(nn.Parameter(torch.ones(1,  dtype=FLOAT)))

    def forward(self, y_stacked):
        """
        y_stacked: float32 tensor of shape (2*M, batch)
                   stacked as [real; imag]
        returns list of intermediate outputs for layer-wise training
        """
        M, N = self.M, self.N
        batch_size = y_stacked.shape[1]

        yreal = y_stacked[:M, :]   # (M, batch)
        yimag = y_stacked[M:, :]   # (M, batch)

        # Initialize ADMM variables
        H = torch.zeros(N, batch_size, dtype=COMPLEX, device=y_stacked.device)
        Z = torch.zeros(N, batch_size, dtype=COMPLEX, device=y_stacked.device)
        U = torch.zeros(N, batch_size, dtype=COMPLEX, device=y_stacked.device)

        # A'*y — computed once, reused every iteration (same as TF version)
        AHY = torch.complex(
            torch.matmul(self.AT, yreal),
            torch.matmul(self.AT, yimag)
        )

        outputs = []  # collect layer outputs for layer-wise training

        for t in range(self.T):
            H_matrix  = self.H_matrices[t] if self.untied else self.H_matrices[0]
            rho       = self.rhos[t]
            theta     = self.thetas[t]
            alpha     = self.alphas[t]

            rho_c   = rho.to(COMPLEX)
            alpha_c = alpha.to(COMPLEX)

            # 1. H-update
            input_H = AHY + rho_c * Z - U
            H = torch.complex(
                torch.matmul(H_matrix, input_H.real),
                torch.matmul(H_matrix, input_H.imag)
            )
            outputs.append(('H-update', t, H))

            # 2. Z-update (soft threshold)
            V = H + U / rho_c
            if self.shrink == 'soft':
                threshold = (theta[0] / rho[0]).to(FLOAT)
                Z = complex_soft_threshold(V, threshold)
            outputs.append(('Z-update', t, Z))

            # 3. U-update
            U = U + alpha_c * rho_c * (H - Z)

        return outputs


# ──────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────
def save_model(model, filename, **kwargs):
    state = {'model_state': model.state_dict()}
    state.update(kwargs)
    torch.save(state, filename)
    print('Saved to', filename)


def load_model(model, filename):
    other = {}
    if os.path.exists(filename):
        state = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(state['model_state'])
        print('Loaded model from', filename)
        other = {k: v for k, v in state.items() if k != 'model_state'}
    return other


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def nmse_loss(xhat, x_true):
    """NMSE loss — identical formula to TF version"""
    error = torch.abs(xhat - x_true) ** 2
    power = torch.abs(x_true)        ** 2
    return torch.mean(torch.sum(error, dim=0)) / torch.mean(torch.sum(power, dim=0))


def stack_complex(arr):
    """Stack complex numpy array as [real; imag] float32"""
    arr = arr.astype(np.complex64)
    return np.concatenate([arr.real, arr.imag], axis=0).astype(np.float32)


def to_gpu(arr):
    return torch.tensor(arr, dtype=FLOAT, device=DEVICE)


def do_training(model, savefile, tc, tm, vc, vm,
                trinit=1e-3, refinements=(.5, .1, .01),
                iv1=10, maxit=100000, better_wait=5000,
                batch_size=1024):

    # Load data
    xt = loadmat(tc)['x'].astype(np.complex64)
    yt = loadmat(tm)['y'].astype(np.complex64)
    xv = loadmat(vc)['x'].astype(np.complex64)
    yv = loadmat(vm)['y'].astype(np.complex64)

    trainingsize = xt.shape[1]

    # ✅ Pre-stack everything once as float32 and move validation to GPU
    yt_stack = to_gpu(stack_complex(yt))   # (2M, train_samples) on GPU
    xt_stack = to_gpu(stack_complex(xt))   # (2N, train_samples) on GPU
    yv_gpu   = to_gpu(stack_complex(yv))   # (2M, val_samples)   on GPU

    N = model.N
    xv_complex = torch.complex(
        to_gpu(xv.real), to_gpu(xv.imag)
    )  # (N, val_samples) on GPU

    # Load saved state
    state     = load_model(model, savefile)
    done      = list(state.get('done',      []))
    log       = str(state.get('log',        ''))
    layernmse = list(state.get('layernmse', []))

    model.to(DEVICE)

    # ──────────────────────────────────────────
    # Build training stages — mirrors TF version structure:
    # layer-wise then refinement, for each ADMM layer output
    # ──────────────────────────────────────────
    training_stages = []

    for t in range(model.T):
        for update_type in ['H-update', 'Z-update']:
            stage_name = 'LADMM-soft {} T={}'.format(update_type, t + 1)

            # Which params to train at this stage (layer-wise)
            if update_type == 'H-update':
                layer_params = (
                    list(model.H_matrices[t].unsqueeze(0) if False else [model.H_matrices[t]]) +
                    [model.rhos[t]]
                )
            else:
                layer_params = [model.thetas[t]]

            # All params up to this layer
            all_params = []
            for tt in range(t + 1):
                if model.untied:
                    all_params.append(model.H_matrices[tt])
                all_params += [model.rhos[tt], model.thetas[tt], model.alphas[tt]]

            training_stages.append({
                'name':         stage_name,
                'layer_params': layer_params,
                'all_params':   all_params,
                'T_output':     t,
                'update_type':  update_type,
            })

            # Refinement stages at each layer
            for idx, fm in enumerate(refinements):
                training_stages.append({
                    'name':         '{} trainrate={}'.format(stage_name, idx),
                    'layer_params': [],          # empty = use all_params
                    'all_params':   all_params,
                    'T_output':     t,
                    'update_type':  update_type,
                })

    # ──────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────
    nmse_dB = None

    for stage in training_stages:
        name         = stage['name']
        layer_params = stage['layer_params']
        all_params   = stage['all_params']
        T_out        = stage['T_output']
        update_type  = stage['update_type']

        if name in done:
            print('Already did {}, skipping.'.format(name))
            continue

        # Which optimizer learning rate to use
        is_refinement = 'trainrate' in name
        lr = trinit
        if is_refinement:
            idx = int(name.split('trainrate=')[-1])
            lr  = refinements[idx] * trinit

        # Which params to optimize
        opt_params = all_params if (is_refinement or not layer_params) else layer_params
        optimizer  = torch.optim.Adam(opt_params, lr=lr)

        if layer_params and not is_refinement:
            print('{} — training {} params'.format(name, len(layer_params)))
        else:
            print('{} — fine tuning {} params'.format(name, len(all_params)))

        nmse_history = []
        model.train()

        for i in range(maxit + 1):
            if i % iv1 == 0:
                # ── Validation ──
                model.eval()
                with torch.no_grad():
                    outputs  = model(yv_gpu)
                    # Find the right output for this stage
                    xhat_val = next(
                        Z for (utype, t, Z) in outputs
                        if t == T_out and utype == update_type
                    )
                    nmse_val = nmse_loss(xhat_val, xv_complex).item()

                nmse_val = round(nmse_val, 5)
                if np.isnan(nmse_val):
                    raise RuntimeError('NMSE is NaN')

                nmse_history = np.append(nmse_history, nmse_val)
                nmse_dB      = 10 * np.log10(nmse_val)
                nmsebest_dB  = 10 * np.log10(nmse_history.min())

                sys.stdout.write(
                    '\ri={:<6d} nmse={:.6f} dB (best={:.6f})'.format(
                        i, nmse_dB, nmsebest_dB))
                sys.stdout.flush()

                if i % (iv1 * 100) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1
                    if age_of_best * iv1 > better_wait:
                        break

                model.train()

            # ── Training step ──
            idx_batch = torch.randint(0, trainingsize, (batch_size,))
            y_batch   = yt_stack[:, idx_batch]   # already on GPU
            x_batch   = torch.complex(
                xt_stack[:N,  idx_batch],
                xt_stack[N:,  idx_batch]
            )

            optimizer.zero_grad()
            outputs = model(y_batch)
            xhat    = next(
                Z for (utype, t, Z) in outputs
                if t == T_out and utype == update_type
            )
            loss = nmse_loss(xhat, x_batch)
            loss.backward()
            optimizer.step()

        done.append(name)
        log = log + '\n{} nmse={:.6f} dB in {} iterations'.format(
            name, nmse_dB, i)

        state['done']      = done
        state['log']       = log
        state['layernmse'] = layernmse
        save_model(model, savefile, **state)

    if nmse_dB is not None:
        layernmse.append(nmse_dB)

    print('\nFinal layer NMSE values: {}'.format(layernmse))
    state['layernmse'] = layernmse
    save_model(model, savefile, **state)

    return model


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    N         = 256
    M         = 128
    type_name = 'UPA'
    shrink    = 'soft'
    MN        = str(N) + str(M)
    SNRrange  = '0to20dB'
    vSNRrange = '1_11dB'
    T         = 6

    A = loadmat('CSmatrix' + MN + '.mat')['A'].astype(np.float32)

    save_network = 'Trained_LADMM_for_SV_{}_{}{}_{}.pt'.format(
        type_name, MN, SNRrange, vSNRrange)
    tc = 'SV_{}_training_channel.mat'.format(type_name)
    tm = 'SV_{}_training_{}_measurements_{}.mat'.format(type_name, M, SNRrange)
    vc = 'SV_{}_validation_channel.mat'.format(type_name)
    vm = 'SV_{}_validation_{}_measurements_{}.mat'.format(type_name, M, vSNRrange)

    print('Building LADMM network with T={} layers...'.format(T))
    model = LADMM(A, T=T, shrink=shrink, untied=True)
    model.to(DEVICE)

    print('Starting training...')
    model = do_training(
        model=model,
        savefile=save_network,
        tc=tc, tm=tm, vc=vc, vm=vm,
        trinit=5e-4,
        refinements=(.5, .2, 0.1, .01),
        maxit=100000,
        batch_size=400
    )
    print('Training complete!')