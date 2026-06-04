#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import pickle
import math
import numpy as np
import os
import random
import sklearn.metrics
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from Algoritmo_klcpd.mmd_util import median_heuristic, batch_mmd2_loss
from Algoritmo_klcpd.data_loader import DataLoader
from Algoritmo_klcpd.optim import Optim


class NetG(nn.Module):
    def __init__(self, args, data):
        super(NetG, self).__init__()
        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = args.RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, self.var_dim)

    # X_p:   batch_size x wnd_dim x var_dim (Encoder input)
    # X_f:   batch_size x wnd_dim x var_dim (Decoder input)
    # h_t:   1 x batch_size x RNN_hid_dim
    # noise: 1 x batch_size x RNN_hid_dim
    def forward(self, X_p, X_f, noise):
        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].zero_()
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft


class NetD(nn.Module):
    def __init__(self, args, data):
        super(NetD, self).__init__()

        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = args.RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec


# Y, L should be numpy array
def valid_epoch(loader, data, netD, batch_size, Y_true, L_true, device, sigma_var):
    netD.eval()
    Y_pred = []
    for inputs in loader.get_batches(data, batch_size, shuffle=False):
        X_p, X_f = inputs[0], inputs[1]
        # Convertir a tensor si es numpy array
        if isinstance(X_p, np.ndarray):
            X_p = torch.from_numpy(X_p)
        if isinstance(X_f, np.ndarray):
            X_f = torch.from_numpy(X_f)
        X_p = X_p.to(device)
        X_f = X_f.to(device)
        batch_size = X_p.size(0)

        X_p_enc, _ = netD(X_p)
        X_f_enc, _ = netD(X_f)
        Y_pred_batch = batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)
        Y_pred.append(Y_pred_batch.detach().cpu().numpy())
    Y_pred = np.concatenate(Y_pred, axis=0)

    L_pred = Y_pred
    
    # Calcular AUC si hay muestras positivas y negativas
    if np.sum(L_true) > 0 and np.sum(L_true) < len(L_true):
        fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, L_pred)
        auc = sklearn.metrics.auc(fp_list, tp_list)
    else:
        # Si no hay variación en labels, usar AUC dummy
        auc = 0.5
    
    eval_dict = {'Y_pred': Y_pred,
                 'L_pred': L_pred,
                 'Y_true': Y_true,
                 'L_true': L_true,
                 'mse': -1, 'mae': -1, 'auc': auc}
    return eval_dict



# ========= Setup input argument =========#
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data_path', type=str, required=True, help='path to data in matlab format')
parser.add_argument('--trn_ratio', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--val_ratio', type=float, default=0.8,help='how much data used for validation')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--no_cuda', action='store_true', help='disable GPU and force CPU execution')
parser.add_argument('--random_seed', type=int, default=1126,help='random seed')

parser.add_argument('--wnd_dim', type=int, required=True, default=10, help='window size (past and future)')
parser.add_argument('--sub_dim', type=int, default=1, help='dimension of subspace embedding')

# RNN hyperparemters
parser.add_argument('--RNN_hid_dim', type=int, default=10, help='number of RNN hidden units')

# optimization
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--max_iter', type=int, default=100, help='max iteration for pretraining RNN')
parser.add_argument('--optim', type=str, default='adam', help='sgd|rmsprop|adam for optimization method')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 regularization)')
parser.add_argument('--momentum', type=float, default=0.0, help='momentum for sgd')
parser.add_argument('--grad_clip', type=float, default=10.0, help='gradient clipping for RNN (both netG and netD)')
parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency per generator update')

# GAN
parser.add_argument('--CRITIC_ITERS', type=int, default=5, help='number of updates for critic per generator')
parser.add_argument('--weight_clip', type=float, default=.1, help='weight clipping for crtic')
parser.add_argument('--lambda_ae', type=float, default=0.001, help='coefficient for the reconstruction loss')
parser.add_argument('--lambda_real', type=float, default=0.1, help='coefficient for the real MMD2 loss')


# save models
parser.add_argument('--save_path', type=str,  default='./exp_simulate/jumpingmean/save_RNN',help='path to save the final model')

def detect_changepoints(time_series, wnd_dim=10, max_iter=500, batch_size=128, 
                        lambda_ae=0.001, lambda_real=0.1, weight_clip=0.1,
                        eval_freq=50, gpu=0, no_cuda=False, 
                        trn_ratio=0.6, val_ratio=0.8, random_seed=1126,
                        verbose=False, return_changepoints=True, 
                        return_scores=False, return_auc=False):
    """
    Detección de puntos de cambio usando KLCPD.
    
    Parámetros:
    -----------
    time_series : np.ndarray
        Array numpy con la serie temporal (shape: (N, D) donde N=muestras, D=dimensiones)
    wnd_dim : int, default=10
        Tamaño de ventana
    max_iter : int, default=2000
        Iteraciones máximas de entrenamiento
    batch_size : int, default=64
        Tamaño de batch
    lambda_ae : float, default=0.001
        Coeficiente reconstrucción
    lambda_real : float, default=0.1
        Coeficiente MMD real
    weight_clip : float, default=0.1
        Clipping de pesos
    eval_freq : int, default=50
        Frecuencia evaluación
    gpu : int, default=0
        GPU a usar
    no_cuda : bool, default=False
        Usar solo CPU
    trn_ratio : float, default=0.6
        Ratio entrenamiento
    val_ratio : float, default=0.8
        Ratio validación
    random_seed : int, default=1126
        Semilla
    verbose : bool, default=False
        Mostrar detalles
    return_changepoints : bool, default=True
        Retornar changepoints
    return_scores : bool, default=False
        Retornar scores
    return_auc : bool, default=False
        Retornar AUC
    
    Retorna:
    --------
    Si solo se solicita un valor, retorna ese valor directamente.
    Si se solicitan múltiples valores, retorna tupla (changepoints, scores, auc) con los solicicitados.
    """
    
    # Convertir a numpy si es necesario
    time_series = np.asarray(time_series, dtype=np.float32)
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
    
    # Crear argumentos
    class Args:
        pass
    
    args = Args()
    args.wnd_dim = wnd_dim
    args.var_dim = time_series.shape[1]
    args.D = time_series.shape[0]
    args.RNN_hid_dim = 10
    args.sub_dim = 1
    args.batch_size = batch_size
    args.max_iter = max_iter
    args.optim = 'adam'
    args.lr = 3e-4
    args.weight_decay = 0.0
    args.momentum = 0.0
    args.grad_clip = 10.0
    args.CRITIC_ITERS = 5
    args.weight_clip = weight_clip
    args.lambda_ae = lambda_ae
    args.lambda_real = lambda_real
    args.eval_freq = eval_freq
    args.trn_ratio = trn_ratio
    args.val_ratio = val_ratio
    args.gpu = gpu
    args.no_cuda = no_cuda
    args.random_seed = random_seed
    args.cuda = False
    
    # ========= Setup device and fix random seed =========#
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    args.cuda = use_cuda
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        if verbose:
            print('Using GPU device', torch.cuda.current_device())
    else:
        if verbose:
            print('Using CPU')
    
    np.random.seed(seed=args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)
    
    cudnn.benchmark = True
    cudnn.enabled = True
    
    # ========= Procesar datos del array numpy =========#
    # Generar ventanas deslizantes
    N = len(time_series)
    windows_p = []  # ventanas pasado
    windows_f = []  # ventanas futuro
    labels = []     # etiquetas (si hay cambio)
    
    for i in range(N - 2 * wnd_dim + 1):
        X_p = time_series[i:i + wnd_dim]
        X_f = time_series[i + wnd_dim:i + 2 * wnd_dim]
        windows_p.append(X_p)
        windows_f.append(X_f)
        # Usar etiquetas dummy (solo para estructura)
        labels.append(0)
    
    windows_p = np.array(windows_p, dtype=np.float32)
    windows_f = np.array(windows_f, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # Splits
    n_samples = len(windows_p)
    trn_idx = int(n_samples * args.trn_ratio)
    val_idx = int(n_samples * args.val_ratio)
    
    trn_p, trn_f = windows_p[:trn_idx], windows_f[:trn_idx]
    val_p, val_f = windows_p[trn_idx:val_idx], windows_f[trn_idx:val_idx]
    tst_p, tst_f = windows_p[val_idx:], windows_f[val_idx:]
    
    # Crear dataset dummy para cálculo de sigma
    class DummyData:
        def __init__(self):
            self.var_dim = args.var_dim
            self.D = args.D
            self.Y_subspace = windows_p[:min(100, len(windows_p))].reshape(-1, args.var_dim * args.wnd_dim)  # Usar primeras muestras, aplanadas a 2D
            self.trn_set = {'Y': torch.from_numpy(windows_p), 'L': torch.from_numpy(labels)}
            self.val_set = {'Y': torch.from_numpy(val_p), 'L': torch.from_numpy(labels[:len(val_p)])}
            self.tst_set = {'Y': torch.from_numpy(tst_p), 'L': torch.from_numpy(labels[:len(tst_p)])}
        
        def get_batches(self, dataset, batch_size, shuffle=False):
            data_p = dataset['Y']
            data_f = dataset.get('F', torch.from_numpy(windows_f[:len(dataset['Y'])]))
            
            indices = np.arange(len(data_p))
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_p = data_p[batch_idx]
                
                # Obtener las ventanas futuro correspondientes
                if len(windows_f) > len(data_p):
                    batch_f = windows_f[batch_idx]
                else:
                    batch_f = windows_f[batch_idx]
                
                yield batch_p, batch_f, None
    
    Data = DummyData()
    
    # Inicializar modelos
    netG = NetG(args, Data)
    netD = NetD(args, Data)
    
    netG.to(device)
    netD.to(device)
    
    if verbose:
        print(f'Data shape: {len(windows_p)} samples, {args.var_dim} dimensions')
    
    one = torch.tensor(1., device=device)
    mone = -one
    
    # ========= Setup optimizers =========#
    optimizerG = Optim(netG.parameters(),
                       args.optim,
                       lr=args.lr,
                       grad_clip=args.grad_clip,
                       weight_decay=args.weight_decay,
                       momentum=args.momentum)
    
    optimizerD = Optim(netD.parameters(),
                       args.optim,
                       lr=args.lr,
                       grad_clip=args.grad_clip,
                       weight_decay=args.weight_decay,
                       momentum=args.momentum)
    
    # sigma
    sigma_list = median_heuristic(Data.Y_subspace, beta=.5)
    sigma_var = torch.tensor(sigma_list, dtype=torch.float32, device=device)
    
    # ========= Training loop =========#
    Y_val = val_p
    L_val = np.zeros(len(val_p))
    Y_tst = tst_p
    L_tst = np.zeros(len(tst_p))
    
    n_batchs = int(math.ceil(len(trn_p) / float(args.batch_size)))
    
    lambda_ae = args.lambda_ae
    lambda_real = args.lambda_real
    gen_iterations = 0
    best_predictions = None
    best_mmd_real = 1e+6
    start_time = time.time()
    
    for epoch in range(1, args.max_iter + 1):
        # Generar batches
        trn_indices = np.arange(len(trn_p))
        np.random.shuffle(trn_indices)
        
        bidx = 0
        while bidx < n_batchs:
            # Update D
            for p in netD.parameters():
                p.requires_grad = True
            
            for diters in range(args.CRITIC_ITERS):
                with torch.no_grad():
                    for p in netD.rnn_enc_layer.parameters():
                        p.clamp_(-args.weight_clip, args.weight_clip)
                
                if bidx >= n_batchs:
                    break
                
                # Get batch
                batch_idx = trn_indices[bidx*args.batch_size:(bidx+1)*args.batch_size]
                X_p = torch.from_numpy(trn_p[batch_idx]).to(device)
                X_f = torch.from_numpy(trn_f[batch_idx]).to(device)
                batch_size = X_p.size(0)
                bidx += 1
                
                # Forward
                X_p_enc, X_p_dec = netD(X_p)
                X_f_enc, X_f_dec = netD(X_f)
                
                noise = torch.randn(1, batch_size, args.RNN_hid_dim, device=X_p.device)
                with torch.no_grad():
                    Y_f = netG(X_p, X_f, noise).detach()
                Y_f_enc, Y_f_dec = netD(Y_f)
                
                D_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, sigma_var)
                mmd2_real = batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)
                
                real_L2_loss = torch.mean((X_f - X_f_dec)**2)
                fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2)
                
                netD.zero_grad()
                lossD = D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()
                lossD.backward(mone)
                optimizerD.step()
            
            # Update G
            for p in netD.parameters():
                p.requires_grad = False
            
            if bidx >= n_batchs:
                break
            
            batch_idx = trn_indices[bidx*args.batch_size:(bidx+1)*args.batch_size]
            X_p = torch.from_numpy(trn_p[batch_idx]).to(device)
            X_f = torch.from_numpy(trn_f[batch_idx]).to(device)
            batch_size = X_p.size(0)
            bidx += 1
            
            X_f_enc, X_f_dec = netD(X_f)
            
            noise = torch.randn(1, batch_size, args.RNN_hid_dim, device=X_p.device)
            Y_f = netG(X_p, X_f, noise)
            Y_f_enc, Y_f_dec = netD(Y_f)
            
            G_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, sigma_var)
            
            netG.zero_grad()
            lossG = G_mmd2.mean()
            lossG.backward(one)
            optimizerG.step()
            
            gen_iterations += 1
            
            if verbose and gen_iterations % args.eval_freq == 0:
                print(f"Epoch {epoch}/{args.max_iter} | Iter {gen_iterations} | D_mmd2: {D_mmd2.mean():.4e} | G_mmd2: {G_mmd2.mean():.4e}")
            
            if gen_iterations % args.eval_freq == 0:
                val_dict = valid_epoch(Data, Data.val_set, netD, args.batch_size, Y_val, L_val, device, sigma_var)
                tst_dict = valid_epoch(Data, Data.tst_set, netD, args.batch_size, Y_tst, L_tst, device, sigma_var)
                
                if mmd2_real.mean().item() < best_mmd_real:
                    best_mmd_real = mmd2_real.mean().item()
                    best_predictions = tst_dict
            
            if mmd2_real.mean().item() < 1e-5:
                if verbose:
                    print("Convergencia alcanzada")
                break
        
        if mmd2_real.mean().item() < 1e-5:
            break
    
    # ========= Procesar resultados =========#
    if best_predictions is None:
        best_predictions = valid_epoch(Data, Data.tst_set, netD, args.batch_size, Y_tst, L_tst, device, sigma_var)
    
    scores = best_predictions['Y_pred']
    
    # Calcular cambios solo si se solicita
    changepoints = None
    if return_changepoints or return_scores or return_auc:
        # Calcular umbral
        L_true = best_predictions['L_true']
        
        # Si hay variación en labels, usar ROC; si no, usar percentil
        if np.sum(L_true) > 0 and np.sum(L_true) < len(L_true):
            fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, scores)
            j_scores = tp_list - fp_list
            best_threshold_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[best_threshold_idx]
        else:
            # Sin labels reales, usar percentil 75
            optimal_threshold = np.percentile(scores, 75)
        
        # Detectar cambios
        changepoints = np.where(scores >= optimal_threshold)[0].tolist()
    
    auc = best_predictions['auc']
    
    # Retornar según lo solicitado
    return_flags = [return_changepoints, return_scores, return_auc]
    n_returns = sum(return_flags)
    
    if n_returns == 0:
        # Si no se solicita nada, retornar changepoints por defecto
        return changepoints
    elif n_returns == 1:
        # Si solo se solicita un valor, retornarlo directamente
        if return_changepoints:
            return changepoints
        elif return_scores:
            return scores
        elif return_auc:
            return auc
    else:
        # Si se solicitan múltiples valores, retornar tupla con los solicicitados
        results = []
        if return_changepoints:
            results.append(changepoints)
        if return_scores:
            results.append(scores)
        if return_auc:
            results.append(auc)
        return tuple(results)


# ========= Script interface (para compatibilidad con línea de comandos) =========#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KLCPD - Change Point Detection')
    parser.add_argument('--data_path', type=str, required=True, help='path to .mat file')
    parser.add_argument('--wnd_dim', type=int, default=10, help='window dimension')
    parser.add_argument('--max_iter', type=int, default=2000, help='max iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lambda_ae', type=float, default=0.001, help='reconstruction loss coefficient')
    parser.add_argument('--lambda_real', type=float, default=0.1, help='MMD real loss coefficient')
    parser.add_argument('--weight_clip', type=float, default=0.1, help='weight clipping')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    
    args = parser.parse_args()
    
    # Load data from .mat file
    try:
        from scipy.io import loadmat
        mat_data = loadmat(args.data_path)
        # Buscar el primer array numérico de tamaño significativo
        data = None
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray) and value.size > 100:
                data = value
                break
        
        if data is None:
            print("Error: No data found in .mat file")
            exit(1)
        
        # Detectar cambios
        results = detect_changepoints(
            time_series=data,
            wnd_dim=args.wnd_dim,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            lambda_ae=args.lambda_ae,
            lambda_real=args.lambda_real,
            weight_clip=args.weight_clip,
            gpu=args.gpu,
            verbose=args.verbose
        )
        
        print(f"\nResultados:")
        print(f"  Puntos detectados: {len(results['changepoints'])}")
        print(f"  AUC: {results['auc']:.6f}")
        print(f"  Tiempo: {results['time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
