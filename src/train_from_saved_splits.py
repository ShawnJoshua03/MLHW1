#!/usr/bin/env python3
"""
My ML HW1 training script
I use the saved splits inside the outputs folder
Run this from the repo root like  python src/train_from_saved_splits.py
"""

#all team members have contributed in equal measure to this effort
#krrish thakku suresh
#Shawn Lasrado

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# basic settings for making experiments reproducible 
SEED = 42
#learning rate
LR = 1e-3
EPOCHS = 4000
#64 beacause of efficiency and tradition
BATCH_SIZE = 64
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
#to reproduce same results
rng = np.random.default_rng(SEED)



def read_y_csv(path):
    df = pd.read_csv(path)
    
    #removing unnamed index coloumn which might be created when saving DF to csv
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cols_lower = [c.lower() for c in df.columns]

    if "quality" in cols_lower:
        real = [c for c in df.columns if c.lower() == "quality"][0]
        s = df[real]
    elif df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        # fallback to headerless read
        s = pd.read_csv(path, header=None).squeeze("columns")
    return pd.to_numeric(s, errors="raise").to_numpy(float)

def load_splits():
    base = Path("outputs")
    if not base.exists():
        raise FileNotFoundError("outputs folder not found")
    # expect typical names
    xtr_p = base / "X_train.csv"
    xva_p = base / "X_val.csv"
    xte_p = base / "X_test.csv"
    ytr_p = base / "y_train.csv"
    yva_p = base / "y_val.csv"
    yte_p = base / "y_test.csv"
    if all(p.exists() for p in [xtr_p, xva_p, xte_p, ytr_p, yva_p, yte_p]):
        Xtr = pd.read_csv(xtr_p)
        Xva = pd.read_csv(xva_p)
        Xte = pd.read_csv(xte_p)
        ytr = read_y_csv(ytr_p)
        yva = read_y_csv(yva_p)
        yte = read_y_csv(yte_p)
    else:
        # try npy
        xtr_p = base / "X_train.npy"
        xva_p = base / "X_val.npy"
        xte_p = base / "X_test.npy"
        ytr_p = base / "y_train.npy"
        yva_p = base / "y_val.npy"
        yte_p = base / "y_test.npy"
        if not all(p.exists() for p in [xtr_p, xva_p, xte_p, ytr_p, yva_p, yte_p]):
            raise FileNotFoundError("could not find CSV or NPY splits")
        Xtr = pd.DataFrame(np.load(xtr_p))
        Xva = pd.DataFrame(np.load(xva_p))
        Xte = pd.DataFrame(np.load(xte_p))
        ytr = np.load(ytr_p).ravel().astype(float)
        yva = np.load(yva_p).ravel().astype(float)
        yte = np.load(yte_p).ravel().astype(float)

    return Xtr, ytr, Xva, yva, Xte, yte

def ensure_bias_and_standardize(Xtr_df, Xva_df, Xte_df):
    # detect bias
    has_bias = ("bias" in Xtr_df.columns) or np.allclose(Xtr_df.iloc[:, 0].to_numpy(), 1.0)
    if has_bias:
        feat_cols = [c for c in Xtr_df.columns if c != "bias"]
        
        #mean
        mu = Xtr_df[feat_cols].mean(axis=0)
        #std
        sigma = Xtr_df[feat_cols].std(axis=0).replace(0.0, 1.0)

        Xtr = Xtr_df.copy()
        Xva = Xva_df.copy()
        Xte = Xte_df.copy()
        Xtr[feat_cols] = (Xtr[feat_cols] - mu) / sigma
        Xva[feat_cols] = (Xva[feat_cols] - mu) / sigma
        Xte[feat_cols] = (Xte[feat_cols] - mu) / sigma

        feature_names = list(Xtr.columns)
    else:
        mu = Xtr_df.mean(axis=0)
        sigma = Xtr_df.std(axis=0).replace(0.0, 1.0)
        Xtr = (Xtr_df - mu) / sigma
        Xva = (Xva_df - mu) / sigma
        Xte = (Xte_df - mu) / sigma

        #creating bias coloumns
        ones_tr = pd.Series(1.0, index=Xtr.index, name="bias")
        ones_va = pd.Series(1.0, index=Xva.index, name="bias")
        ones_te = pd.Series(1.0, index=Xte.index, name="bias")
        
        Xtr = pd.concat([ones_tr, Xtr], axis=1)
        Xva = pd.concat([ones_va, Xva], axis=1)
        Xte = pd.concat([ones_te, Xte], axis=1)
        feature_names = list(Xtr.columns)
    return Xtr.to_numpy(float), Xva.to_numpy(float), Xte.to_numpy(float), feature_names

#loss function
def mse(y, yhat):
    e = yhat - y
    return float(np.mean(e * e))

#matrix multiplication of multiplying matrix X with weights w
def predict(X, w):
    return X @ w

def grad_mse(X, y, w):
    m = X.shape[0]
    return (2.0 / m) * (X.T @ (X @ w - y))

def batch_gd(X, y, lr=LR, epochs=EPOCHS, reg=None, alpha=0.0, reg_mask=None):
    m, n = X.shape
    w = np.zeros(n, dtype=float)

    if reg_mask is None:
        reg_mask = np.ones(n, dtype=float)
    hist = []

    for ep in range(epochs):
        g = grad_mse(X, y, w)
        if reg == "l2" and alpha > 0.0:
            g = g + 2.0 * alpha * (w * reg_mask)
        if reg == "l1" and alpha > 0.0:
            g = g + 2.0 * alpha * (np.sign(w) * reg_mask)
        w = w - lr * g
        if ep % 200 == 0 or ep == epochs - 1:
            hist.append(mse(y, predict(X, w)))
    return w, hist

def minibatch_gd(X, y, lr=LR, epochs=EPOCHS, batch=BATCH_SIZE, reg=None, alpha=0.0, reg_mask=None):
    m, n = X.shape
    w = np.zeros(n, dtype=float)

    if reg_mask is None:
        reg_mask = np.ones(n, dtype=float)
    steps = max(1, m // batch)
    hist = []

    for ep in range(epochs):
        idx = rng.permutation(m)
        Xp = X[idx]
        yp = y[idx]
        for s in range(steps):
            a = s * batch
            b = min(m, a + batch)
            Xb = Xp[a:b]
            yb = yp[a:b]
            g = grad_mse(Xb, yb, w)
            if reg == "l2" and alpha > 0.0:
                g = g + 2.0 * alpha * (w * reg_mask)
            if reg == "l1" and alpha > 0.0:
                g = g + 2.0 * alpha * (np.sign(w) * reg_mask)
            w = w - lr * g
        if ep % 200 == 0 or ep == epochs - 1:
            hist.append(mse(y, predict(X, w)))
    return w, hist

def choose_alpha(train_fn, Xtr, ytr, Xva, yva, alphas, reg, reg_mask):
    best_a = None
    best_v = float("inf")
    best_w = None
    rows = []

    for a in alphas:
        w, _ = train_fn(Xtr, ytr, reg=reg, alpha=a, reg_mask=reg_mask)
        val = mse(yva, predict(Xva, w))
        rows.append({"alpha": a, "val_mse": val})
        if val < best_v:
            best_v = val
            best_a = a
            best_w = w
    return best_a, best_w, pd.DataFrame(rows)

def drop_smallest(w, names):
    wa = np.abs(w)
    j = 1 + int(np.argmin(wa[1:]))  # skip bias at zero
    keep = [k for k in range(len(w)) if k != j]
    return j, names[j], np.array(keep, dtype=int)

def largest_feature(w, names):
    wa = np.abs(w)
    j = 1 + int(np.argmax(wa[1:]))
    return j, names[j]

def plot_line(X, y, w, j, name, title, path):
    x = X[:, j]
    w0 = w[0]
    wj = w[j]
    xs = np.linspace(x.min(), x.max(), 200)
    ys = w0 + wj * xs
    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.6)
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel(name)
    plt.ylabel("quality")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()

def main():
    print("loading splits")
    Xtr_df, ytr, Xva_df, yva, Xte_df, yte = load_splits()

    print("standardizing and adding bias if needed")
    Xtr, Xva, Xte, names = ensure_bias_and_standardize(Xtr_df, Xva_df, Xte_df)

    #building reg mask, not penalizing bias 
    reg_mask = np.ones(len(names), dtype=float)
    if names[0] == "bias":
        reg_mask[0] = 0.0

    out = Path("model_outputs")
    out.mkdir(exist_ok=True, parents=True)

    print("training plain batch")
    w_plain, h_plain = batch_gd(Xtr, ytr)
    res = {}
    res["plain"] = {
        "train_mse": mse(ytr, predict(Xtr, w_plain)),
        "valid_mse": mse(yva, predict(Xva, w_plain)),
        "test_mse": mse(yte, predict(Xte, w_plain)),
    }
    j_big, name_big = largest_feature(w_plain, names)
    plot_line(Xtr, , w_plain, j_big, name_big, f"batch plain feature {name_big}", str(out / "plot_batch_plain.png"))

    print("training mini batch plain")
    w_mini, h_mini = minibatch_gd(Xtr, ytr)
    res["minibatch_plain"] = {
        "train_mse": mse(ytr, predict(Xtr, w_mini)),
        "valid_mse": mse(yva, predict(Xva, w_mini)),
        "test_mse": mse(yte, predict(Xte, w_mini)),
    }

    print("training with l2")
    a2, w2, tab2 = choose_alpha(batch_gd, Xtr, ytr, Xva, yva, ALPHAS, reg="l2", reg_mask=reg_mask)
    res["l2"] = {
        "alpha": a2,
        "train_mse": mse(ytr, predict(Xtr, w2)),
        "valid_mse": mse(yva, predict(Xva, w2)),
        "test_mse": mse(yte, predict(Xte, w2)),
    }
    j2, name2 = largest_feature(w2, names)
    plot_line(Xtr, ytr, w2, j2, name2, f"batch with l2 alpha {a2}", str(out / "plot_batch_l2.png"))

    print("dropping smallest by l2 then retrain plain")
    jdrop2, drop2, keep2 = drop_smallest(w2, names)
    Xtr2 = Xtr[:, keep2]
    Xva2 = Xva[:, keep2]
    Xte2 = Xte[:, keep2]
    names2 = [names[k] for k in keep2]
    w_plain2, _ = batch_gd(Xtr2, ytr)
    res["plain_after_l2_drop"] = {
        "dropped_feature": drop2,
        "train_mse": mse(ytr, predict(Xtr2, w_plain2)),
        "valid_mse": mse(yva, predict(Xva2, w_plain2)),
        "test_mse": mse(yte, predict(Xte2, w_plain2)),
    }
    j2b, name2b = largest_feature(w_plain2, names2)
    plot_line(Xtr2, ytr, w_plain2, j2b, name2b, "batch plain after l2 drop", str(out / "plot_after_l2_drop.png"))

    print("training with l1")
    a1, w1, tab1 = choose_alpha(batch_gd, Xtr, ytr, Xva, yva, ALPHAS, reg="l1", reg_mask=reg_mask)
    res["l1"] = {
        "alpha": a1,
        "train_mse": mse(ytr, predict(Xtr, w1)),
        "valid_mse": mse(yva, predict(Xva, w1)),
        "test_mse": mse(yte, predict(Xte, w1)),
    }
    j1, name1 = largest_feature(w1, names)
    plot_line(Xtr, ytr, w1, j1, name1, f"batch with l1 alpha {a1}", str(out / "plot_batch_l1.png"))

    print("dropping by l1 then retrain plain")
    w1a = np.array(w1)
    zero_idx = np.where(np.isclose(w1a, 0.0, atol=1e-8))[0]
    zero_idx = zero_idx[zero_idx != 0]  # do not drop bias
    if zero_idx.size > 0:
        keep1 = np.array([k for k in range(len(w1a)) if k not in zero_idx], dtype=int)
        drop1 = [names[k] for k in zero_idx]
    else:
        jdrop1, drop1_name, keep1 = drop_smallest(w1a, names)
        drop1 = [drop1_name]
    Xtr1 = Xtr[:, keep1]
    Xva1 = Xva[:, keep1]
    Xte1 = Xte[:, keep1]
    names1 = [names[k] for k in keep1]
    w_plain1, _ = batch_gd(Xtr1, ytr)
    res["plain_after_l1_drop"] = {
        "dropped_feature": drop1,
        "train_mse": mse(ytr, predict(Xtr1, w_plain1)),
        "valid_mse": mse(yva, predict(Xva1, w_plain1)),
        "test_mse": mse(yte, predict(Xte1, w_plain1)),
    }
    j1b, name1b = largest_feature(w_plain1, names1)
    plot_line(Xtr1, ytr, w_plain1, j1b, name1b, "batch plain after l1 drop", str(out / "plot_after_l1_drop.png"))

    # write results
    with open(out / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open(out / "results_summary.txt", "w", encoding="utf-8") as f:
        for k, v in res.items():
            f.write(k + "  " + " | ".join(f"{kk} {vv}" for kk, vv in v.items()) + "\n")

    print("done  saved files to model_outputs")

if __name__ == "__main__":
    np.random.seed(SEED)
    main()
