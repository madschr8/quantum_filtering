import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from help_functions import (apply_cptp, BCE_loss, normalize_kraus,
                            rho_to_real_vec, to_complex_4x2x2)


class LSTMKraus(nn.Module):
    """LSTM that emits a CPTP Kraus map per step from the measurement record.

    The LSTM consumes (J(t), task_one_hot). At every time step a task-conditional
    head produces 32 real Kraus parameters and `normalize_kraus` ensures CPTP by the QR; 
    the projected K_n updates rho deterministically.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        lstm_in = 1 + 3   # J(t) + task one-hot
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.1)

        kraus_in = hidden_dim + 8 + 1   # h_t + rho_vec + J(t)
        self.fc_kraus_z = self._make_kraus_head(kraus_in, hidden_dim)
        self.fc_kraus_x = self._make_kraus_head(kraus_in, hidden_dim)
        self.fc_kraus_y = self._make_kraus_head(kraus_in, hidden_dim)

    @staticmethod
    def _make_kraus_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 32),
        )
        for layer in head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
        # Bias K_0 toward identity at init so the initial map is near no-op.
        head[-1].bias.data[:4] = torch.tensor([1., 0., 0., 1.])
        return head

    def _kraus_step(self, h, rho, task_oh, J_t):
        feat       = torch.cat([h, rho_to_real_vec(rho), J_t], dim=-1)
        params_all = torch.stack([self.fc_kraus_z(feat),
                                  self.fc_kraus_x(feat),
                                  self.fc_kraus_y(feat)], dim=1)
        params     = (params_all * task_oh.unsqueeze(-1)).sum(dim=1)
        K          = normalize_kraus(to_complex_4x2x2(params))
        return K, apply_cptp(K, rho)

    def forward(self, J_tilde, rho0, task_id, return_kraus: bool = False):
        B, T    = J_tilde.shape
        task_oh = F.one_hot(task_id, num_classes=3).float()

        task_seq = task_oh.unsqueeze(1).expand(B, T, 3)
        J_seq    = torch.cat([J_tilde.unsqueeze(-1), task_seq], dim=-1)

        if rho0.dim() == 2:
            rho = rho0.unsqueeze(0).expand(B, -1, -1).clone().to(torch.complex64)
        else:
            rho = rho0.to(torch.complex64).clone()

        h_seq, _ = self.lstm(J_seq)

        rho_list = []
        K_list   = [] if return_kraus else None
        for t in range(T):
            K, rho = self._kraus_step(h_seq[:, t, :], rho, task_oh, J_tilde[:, t:t+1])
            rho_list.append(rho)
            if return_kraus:
                K_list.append(K)

        # Measured-basis end-population for each task, then pick the active one.
        P_end_z = rho[:, 0, 0].real
        P_end_x = (1.0 + 2.0 * rho[:, 0, 1].real) / 2.0
        P_end_y = (1.0 - 2.0 * rho[:, 0, 1].imag) / 2.0
        P_end   = (torch.stack([P_end_z, P_end_x, P_end_y], dim=1) * task_oh).sum(1)
        P_end   = P_end.clamp(1e-6, 1 - 1e-6)

        rho_seq = torch.stack(rho_list, dim=1)
        if return_kraus:
            return rho_seq, P_end, torch.stack(K_list, dim=1)
        return rho_seq, P_end


def train(model,
          J_train, J_val,
          y_train, y_val,
          task_train, task_val,
          rho0,
          epochs       = 20,
          batch        = 64,
          lr_max       = 1e-3,
          lr_min       = 1e-5,
          print_every  = 1,
          val_every    = 1,
          weight_decay = 1e-4):
    """Train LSTMKraus with cosine-annealed AdamW, gradient clipping and BCE on P_end."""
    train_dl = DataLoader(TensorDataset(J_train, y_train, task_train),
                          batch_size=batch, shuffle=True, drop_last=False)
    val_dl   = DataLoader(TensorDataset(J_val, y_val, task_val),
                          batch_size=batch, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_dl), eta_min=lr_min)

    loss_train, loss_val, lr_history, grad_norm_history = [], [], [], []
    current_loss_val = None

    for epoch in range(epochs):
        model.train()
        epoch_losses, epoch_lrs, epoch_gnorms = [], [], []
        for J_b, y_b, task_b in train_dl:
            rho0_b = rho0.unsqueeze(0).expand(len(J_b), -1, -1).clone()
            _, P_end = model(J_b, rho0_b, task_b)
            loss = BCE_loss(P_end, y_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_lrs.append(scheduler.get_last_lr()[0])
            epoch_gnorms.append(float(gnorm))

        loss_train.append(float(np.mean(epoch_losses)))
        lr_history.append(float(np.mean(epoch_lrs)))
        grad_norm_history.append(float(np.mean(epoch_gnorms)))

        if epoch % val_every == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for J_b, y_b, task_b in val_dl:
                    rho0_b = rho0.unsqueeze(0).expand(len(J_b), -1, -1).clone()
                    _, P_v = model(J_b, rho0_b, task_b)
                    val_losses.append(BCE_loss(P_v, y_b).item())
            current_loss_val = float(np.mean(val_losses))
            loss_val.append((epoch, current_loss_val))

        if epoch % print_every == 0 and current_loss_val is not None:
            print(f"Epoch {epoch:4d} | train {loss_train[-1]:.4f} | val {current_loss_val:.4f} "
                  f"| lr {lr_history[-1]:.2e} | |g| {grad_norm_history[-1]:.2f}")

    return {'train': loss_train, 'val': loss_val,
            'lr': lr_history, 'grad_norm': grad_norm_history}
