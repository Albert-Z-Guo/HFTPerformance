"""
Real-data validation of LOB exit strategy model.
Downloads GC=F (Gold Futures) 1-minute OHLCV from Yahoo Finance and runs:
  1. Price-process calibration (σ_daily, σ_tick, intraday vol pattern)
  2. Spread distribution (Gamma fit to H-L range)
  3. Hawkes clustering (volume ACF, ratio heuristic Spearman, burst latency)
  4. Kalman vs EMA on rolling 30-min trade windows
  5. HJB three-region policy with real spreads

Usage:
    python3 docs/real_data_analysis.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, gamma as gamma_dist

# ── Download ──────────────────────────────────────────────────────────────────
import yfinance as yf
gc = yf.download("GC=F", period="5d", interval="1m", progress=False)
gc.columns = [c[0] for c in gc.columns]
gc = gc.dropna()
gc["spread"] = (gc["High"] - gc["Low"]) / 2.0
gc["ret"]    = gc["Close"].pct_change()
prices  = gc["Close"].values.astype(float)
spreads = gc["spread"].values.astype(float)
N       = len(prices)
mid     = np.mean(prices)
print(f"GC=F  N={N} bars  range={gc.index[0].date()}–{gc.index[-1].date()}  price=${mid:.1f}")

# ── 1. Price process ──────────────────────────────────────────────────────────
sigma_1m   = gc["ret"].std()
sigma_daily= sigma_1m * np.sqrt(6.5 * 60)
sigma_sec  = sigma_1m * mid / np.sqrt(60.0)
sigma_10ms = sigma_sec * np.sqrt(0.01)
print(f"\n[1] σ_daily={sigma_daily*100:.3f}%=${sigma_daily*mid:.2f}  σ_10ms=${sigma_10ms:.5f}")

# ── 2. Spread ─────────────────────────────────────────────────────────────────
sp = gc["spread"][gc["spread"] > 0]
sp_mean = sp.mean();  sp_cv = sp.std() / sp_mean
shape, loc, scale = gamma_dist.fit(sp.values, floc=0.01)
print(f"[2] Spread mean=${sp_mean:.3f}  CV={sp_cv:.3f}  Gamma(k={shape:.2f},θ={scale:.3f})")

# ── 3. Volume ACF → Hawkes ────────────────────────────────────────────────────
vol    = gc["Volume"].values.astype(float)
acf1   = np.corrcoef(vol[:-1], vol[1:])[0, 1]
beta_m = -np.log(max(acf1, 1e-4))           # /min
MU_vol = np.mean(vol[vol > 0])
ALPHA  = MU_vol * acf1

# Recursive Hawkes on volume series
R_h = np.zeros(N); lam = np.zeros(N)
for i in range(1, N):
    R_h[i] = (R_h[i-1] + ALPHA * vol[i-1] / MU_vol) * np.exp(-beta_m)
    lam[i]  = MU_vol + R_h[i]

def sliding_ratio(v, ws, wl):
    r = np.ones(len(v))
    for i in range(wl, len(v)):
        r[i] = np.mean(v[max(0,i-ws):i]) / (np.mean(v[max(0,i-wl):i]) + 1e-10)
    return r

rm  = sliding_ratio(vol, ws=1,  wl=10)
rmm = sliding_ratio(vol, ws=10, wl=100)
idx = np.arange(100, N, 10)
r_m,  _ = spearmanr(lam[idx], rm[idx])
r_mm, _ = spearmanr(lam[idx], rmm[idx])

bursts = np.where((lam[1:] > 2*MU_vol) & (lam[:-1] <= 2*MU_vol))[0] + 1
lags_m, lags_mm = [], []
for bi in bursts:
    for j in range(bi, min(N, bi+50)):
        if rm[j] > 1.5:  lags_m.append(j-bi);  break
    else: lags_m.append(50)
    for j in range(bi, min(N, bi+200)):
        if rmm[j] > 1.5: lags_mm.append(j-bi); break
    else: lags_mm.append(200)

print(f"[3] ACF(1)={acf1:.4f}  β={beta_m:.3f}/min  Spearman r={r_m:.4f}")
print(f"    Bursts={len(bursts)}  lag_matched={np.mean(lags_m):.2f}min"
      f"  lag_mismatch={np.mean(lags_mm):.2f}min  ({np.mean(lags_mm)/max(np.mean(lags_m),0.1):.0f}× speedup)")

# ── 4. Kalman vs EMA — rolling 30-min windows ─────────────────────────────────
Q_1min = (sigma_1m * mid)**2;  R_1min = (sp_mean/2)**2
snr_1m  = Q_1min / R_1min
K0_1m   = (np.sqrt(snr_1m**2 + 4*snr_1m) - snr_1m) / 2.0
TICK_SP = 0.15   # GC ECN bid-ask spread (1.5 ticks × $0.10)
snr_tick= sigma_10ms**2 / (TICK_SP/2)**2
K0_tick = (np.sqrt(snr_tick**2 + 4*snr_tick) - snr_tick) / 2.0
print(f"[4] SNR_1min={snr_1m:.4f} (price-dominated)  SNR_tick={snr_tick:.5f} (spread-dominated)")
print(f"    K0_tick={K0_tick:.4f}")

WINDOW = 30;  q75_sp = np.quantile(spreads[spreads>0], 0.75)
q_p2 = (sigma_1m*mid)**2 / 60.0;  q_d2 = 1e-8
kf_fps, ema_fps, kf_fps_w, ema_fps_w = [], [], [], []
for w in range((N - WINDOW) // WINDOW):
    i0, i1 = w*WINDOW, w*WINDOW+WINDOW
    p_win, sp_win = prices[i0:i1], spreads[i0:i1]
    entry = p_win[0];  mn_sp = np.mean(sp_win[sp_win>0]) if any(sp_win>0) else 1.0
    xp, xd = entry, 0.0
    P = np.array([[1.0,0.0],[0.0,0.01]])
    ema_p = entry;  ema_atr = mn_sp
    kf_fp=0; ema_fp=0; n_null=0; kf_fp_w=0; ema_fp_w=0; n_null_w=0
    for j in range(1, WINDOW):
        z = p_win[j]; sp_j = max(sp_win[j], 0.05); dt = 60.0
        xp_pred = xp + xd*dt
        P00 = P[0,0]+dt*(P[0,1]+P[1,0])+dt*dt*P[1,1]+q_p2*dt
        P01 = P[0,1]+dt*P[1,1]; P10=P[1,0]+dt*P[1,1]; P11=P[1,1]+q_d2*dt
        half=sp_j*0.5; R_k=half*half+1e-30; S=P00+R_k
        K0=P00/S; K1=P10/S; inn=z-xp_pred
        xp=xp_pred+K0*inn; xd=xd+K1*inn
        P=np.array([[(1-K0)*P00,(1-K0)*P01],[P10-K1*P00,max(P11-K1*P01,0)]]); P[0,0]=max(P[0,0],0)
        ema_p=K0_1m*z+(1-K0_1m)*ema_p; ema_atr=0.1*abs(p_win[j]-p_win[j-1])+0.9*ema_atr
        adv=abs(p_win[j]-entry)>2.0*mn_sp
        sig_post=np.sqrt(max(P[0,0],0)+R_k)
        kf_sig=abs(xp-entry)/(sig_post+1e-30)>2.0; ema_sig=abs(ema_p-entry)>0.5*ema_atr
        if not adv:
            n_null+=1; kf_fp+=kf_sig; ema_fp+=ema_sig
            if sp_j>q75_sp: n_null_w+=1; kf_fp_w+=kf_sig; ema_fp_w+=ema_sig
    if n_null>0:   kf_fps.append(kf_fp/n_null); ema_fps.append(ema_fp/n_null)
    if n_null_w>0: kf_fps_w.append(kf_fp_w/n_null_w); ema_fps_w.append(ema_fp_w/n_null_w)

kf_fpr=np.mean(kf_fps); ema_fpr=np.mean(ema_fps)
kf_fpr_w=np.mean(kf_fps_w); ema_fpr_w=np.mean(ema_fps_w)
print(f"    Kalman FPR={kf_fpr:.4f}  EMA FPR={ema_fpr:.4f}  reduction={(1-kf_fpr/ema_fpr)*100:.1f}%")
print(f"    Wide-spread: Kalman={kf_fpr_w:.4f}  EMA={ema_fpr_w:.4f}  reduction={(1-kf_fpr_w/ema_fpr_w)*100:.1f}%")

# ── 5. HJB three-region ───────────────────────────────────────────────────────
np.random.seed(42)
ret_sign = np.sign(np.diff(prices, prepend=prices[0]))
imb = np.zeros(N)
for i in range(1,N): imb[i]=np.clip(0.8*imb[i-1]+0.25*ret_sign[i]+0.1*np.random.randn(),-1,1)
e_gain = 2*0.10; hs_tick=TICK_SP/2; q_min=hs_tick/e_gain
Q_ref=0.40; KAPPA=0.15; A=0.30
n_never=sum(1 for i in range(N) if Q_ref<q_min)
n_as   =sum(1 for i in range(N) if q_min<=Q_ref<abs(KAPPA*imb[i]/(2*A)))
n_hjb  =N-n_never-n_as
print(f"[5] q_min={q_min:.4f} (2-tick target)  Q_ref={Q_ref}")
print(f"    Never={100*n_never/N:.1f}%  AS={100*n_as/N:.1f}%  HJB={100*n_hjb/N:.1f}%")
