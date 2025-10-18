# ==== Logging & Plotting helpers ====
import time, csv
from collections import deque
import matplotlib.pyplot as plt

class EMA:
    """Exponential Moving Average for smoothing iteration curves."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        self.v = x if self.v is None else (self.alpha * x + (1 - self.alpha) * self.v)
        return self.v

class TrainVisualizer:
    def __init__(self, outdir, it_plot_every=100, ema_alpha=0.1):
        self.outdir = Path(outdir)
        (self.outdir / "plots").mkdir(parents=True, exist_ok=True)
        (self.outdir / "csv").mkdir(parents=True, exist_ok=True)

        # per-iteration logs
        self.it_idx = []
        self.g_loss = []; self.d_loss = []; self.gan_loss = []; self.aux_loss = []; self.fm_loss = []
        self.ema_g = EMA(ema_alpha); self.ema_d = EMA(ema_alpha)

        # per-epoch logs
        self.epoch = []
        self.val_iou = []; self.val_dice = []; self.val_loss = []; self.lr_hist = []

        self.it_plot_every = it_plot_every
        self._it_last_plot = time.time()

        # CSV writers (lazy)
        self._it_csv = None
        self._ep_csv = None

    # ---------- iteration level ----------
    def log_iter(self, i, g, d, gan, aux, fm):
        self.it_idx.append(i)
        self.g_loss.append(g); self.d_loss.append(d)
        self.gan_loss.append(gan); self.aux_loss.append(aux); self.fm_loss.append(fm)

        # write row to CSV
        if self._it_csv is None:
            self._it_csv = open(self.outdir/'csv/train_iters.csv', 'w', newline='', encoding='utf-8')
            self._it_writer = csv.writer(self._it_csv)
            self._it_writer.writerow(['iter','G','D','GAN','AUX','FM','G_EMA','D_EMA'])
        ge = self.ema_g.update(g); de = self.ema_d.update(d)
        self._it_writer.writerow([i, g, d, gan, aux, fm, ge, de])

        # periodic plot
        if (len(self.it_idx) % self.it_plot_every) == 0:
            self.plot_iteration_curves()

    def plot_iteration_curves(self):
        if not self.it_idx: return
        its = self.it_idx
        g = self.g_loss; d = self.d_loss; gan = self.gan_loss; aux = self.aux_loss; fm = self.fm_loss

        # Compute EMA for nice curves
        def ema_arr(arr, alpha=0.1):
            e=None; out=[]
            for x in arr:
                e = x if e is None else (alpha*x + (1-alpha)*e)
                out.append(e)
            return out
        g_ema = ema_arr(g); d_ema = ema_arr(d)

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(2,1,1)
        ax.plot(its, g,  label='G');  ax.plot(its, d,  label='D')
        ax.plot(its, g_ema, '--', label='G_EMA'); ax.plot(its, d_ema, '--', label='D_EMA')
        ax.set_title('Train losses (iteration)'); ax.set_xlabel('iteration'); ax.set_ylabel('loss'); ax.legend(); ax.grid(True)

        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(its, gan, label='GAN'); ax2.plot(its, aux, label='AUX'); ax2.plot(its, fm, label='FM')
        ax2.set_xlabel('iteration'); ax2.set_ylabel('loss'); ax2.legend(); ax2.grid(True)

        fig.tight_layout()
        fig.savefig(self.outdir/'plots/train_iter_losses.png', dpi=140)
        plt.close(fig)

    # ---------- epoch level ----------
    def log_epoch(self, ep, val_loss, val_iou, val_dice, lr):
        self.epoch.append(ep); self.val_loss.append(val_loss)
        self.val_iou.append(val_iou); self.val_dice.append(val_dice); self.lr_hist.append(lr)

        if self._ep_csv is None:
            self._ep_csv = open(self.outdir/'csv/val_epochs.csv', 'w', newline='', encoding='utf-8')
            self._ep_writer = csv.writer(self._ep_csv)
            self._ep_writer.writerow(['epoch','val_loss','IoU_fg','Dice_fg','lr'])
        self._ep_writer.writerow([ep, val_loss, val_iou, val_dice, lr])
        self.plot_epoch_curves()

    def plot_epoch_curves(self):
        if not self.epoch: return
        ep = self.epoch
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(ep, self.val_loss, '-o', label='val_loss'); ax.grid(True); ax.legend(); ax.set_xlabel('epoch')
        ax = fig.add_subplot(1,2,2)
        ax.plot(ep, self.val_iou,  '-o', label='IoU_fg')
        ax.plot(ep, self.val_dice, '-o', label='Dice_fg')
        ax.set_xlabel('epoch'); ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.outdir/'plots/val_epoch_metrics.png', dpi=140)
        plt.close(fig)

    def close(self):
        if self._it_csv: self._it_csv.close()
        if self._ep_csv: self._ep_csv.close()
