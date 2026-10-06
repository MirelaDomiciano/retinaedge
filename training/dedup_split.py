"""
Deduplicação + split estratificado por classe e origem (APTOS; depois APTOS + EyePACS).

Este arquivo só define funções. No notebook:
    %run /content/drive/MyDrive/tcc/code/dedup_split.py
e chame as funções célula a célula.
"""
import hashlib
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ---------------- Configuração padrão (ajustável no notebook) ----------------
SOURCES = [
    # (nome, csv de rótulos, pasta de imagens, coluna id, coluna rótulo, extensão)
    ("aptos", "/content/data/aptos/train.csv", "/content/data/aptos/train_images", "id_code", "diagnosis", ".png"),
    # ("eyepacs", "/content/data/eyepacs/trainLabels.csv", "/content/data/eyepacs/train", "image", "level", ".jpeg"),
]
HASH_SIZE = 8         # 8 -> hash de 64 bits; 16 -> 256 bits (mais discriminativo; thresholds ~4x maiores)
PHASH_THRESH = 2      # distância de Hamming máxima para considerar quase-duplicata (com pHash, sempre par)
SPLITS = (0.70, 0.15, 0.15)
SEED = 42
OUT_DIR = Path("/content/drive/MyDrive/tcc/data")

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_sources():
    dfs = []
    for name, csv, img_dir, id_col, lbl_col, ext in SOURCES:
        d = pd.read_csv(csv)
        dfs.append(pd.DataFrame({
            "path": [str(Path(img_dir) / f"{i}{ext}") for i in d[id_col]],
            "label": d[lbl_col].astype(int),
            "source": name,
        }))
    df = pd.concat(dfs, ignore_index=True)
    exists = df["path"].map(lambda p: Path(p).exists())
    if (~exists).any():
        print(f"[aviso] {(~exists).sum()} arquivos do CSV não encontrados — removidos")
    return df[exists].reset_index(drop=True)


def _crop_retina(im, tol=10):
    """Recorta a caixa que contém a retina, removendo as bordas pretas.
    As bordas variam entre fotos e dominam o hash, fazendo pacientes diferentes parecerem iguais."""
    gray = np.asarray(im.convert("L"))
    ys, xs = np.nonzero(gray > tol)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def compute_hashes(df, cache_dir):
    """MD5 (cópia exata) + pHash da retina recortada (quase-duplicata). Usa cache por HASH_SIZE."""
    cache = Path(cache_dir) / f"hashes_crop_{HASH_SIZE}.csv"
    if cache.exists():
        cached = pd.read_csv(cache)
        if set(df["path"]) <= set(cached["path"]):
            print(f"[cache] hashes reaproveitados ({cache.name})")
            return df.merge(cached[["path", "md5", "phash_hex"]], on="path", how="left")
    md5s, phs = [], []
    for p in tqdm(df["path"], desc="hashing"):
        with open(p, "rb") as f:
            md5s.append(hashlib.md5(f.read()).hexdigest())
        with Image.open(p) as im:
            phs.append(str(imagehash.phash(_crop_retina(im).convert("L"), hash_size=HASH_SIZE)))
    df = df.assign(md5=md5s, phash_hex=phs)
    df[["path", "md5", "phash_hex"]].to_csv(cache, index=False)
    return df


def _hash_bytes(df):
    bits = np.stack([imagehash.hex_to_hash(h).hash.flatten() for h in df["phash_hex"]])
    return np.packbits(bits, axis=1)


def near_pairs(df, max_dist=10):
    """Todos os pares de imagens com distância <= max_dist (comparação direta, sem encadear).
    Calcule uma vez com max_dist folgado; depois filtre por PHASH_THRESH sem recalcular."""
    B = _hash_bytes(df)
    I, J, D = [], [], []
    for i in tqdm(range(len(B) - 1), desc="comparando"):
        d = _POPCOUNT[np.bitwise_xor(B[i], B[i + 1:])].sum(axis=1).astype(np.int64)
        idx = np.nonzero(d <= max_dist)[0]
        I.append(np.full(len(idx), i)); J.append(i + 1 + idx); D.append(d[idx])
    return pd.DataFrame({"i": np.concatenate(I), "j": np.concatenate(J), "dist": np.concatenate(D)})


def deduplicate(df, pairs):
    """Remove quase-duplicatas sem encadeamento. `pairs` já filtrado por PHASH_THRESH.

    1. Pares com rótulos diferentes: as duas imagens saem (rótulo não confiável).
    2. Demais: percorre as imagens em ordem e mantém cada uma só se ela não for
       quase-duplicata de nenhuma já mantida. Garante que nenhum par mantido fica
       a <= threshold (nada vaza entre train e test) e que uma imagem só é removida
       por ser parecida com uma imagem que ficou — não por "corrente" A~B~C.
    """
    labels = df["label"].to_numpy()
    pi, pj = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    conflict = labels[pi] != labels[pj]
    conflicted = set(pi[conflict]) | set(pj[conflict])

    neighbors = defaultdict(set)
    for a, b in zip(pi, pj):
        neighbors[a].add(b); neighbors[b].add(a)

    kept, removed_dup = set(), set()
    for i in range(len(df)):
        if i in conflicted:
            continue
        if neighbors[i] & kept:
            removed_dup.add(i)
        else:
            kept.add(i)

    report = {
        "imagens": len(df),
        "duplicatas exatas (md5)": int(df["md5"].duplicated().sum()),
        "pares de quase-duplicatas": len(pairs),
        "pares com rótulo conflitante": int(conflict.sum()),
        "imagens removidas por rótulo conflitante": len(conflicted),
        "imagens removidas como duplicata": len(removed_dup),
        "imagens finais": len(kept),
    }
    return df.iloc[sorted(kept)].reset_index(drop=True), report


def _display(im, size=256):
    """Retina recortada e reduzida sem distorcer a proporção (só para visualização)."""
    im = _crop_retina(im).convert("RGB")
    im.thumbnail((size, size))
    return im


def show_near_pairs(df, pairs, dist=None, n=6, seed=0):
    """Mostra pares lado a lado (origem, grau, distância) para validar o threshold.
    dist=None: pares de qualquer distância; dist=k: só pares exatamente nessa distância."""
    import matplotlib.pyplot as plt
    sel = pairs if dist is None else pairs[pairs["dist"] == dist]
    if sel.empty:
        print("nenhum par nessa distância")
        return
    for _, r in sel.sample(min(n, len(sel)), random_state=seed).iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for ax, k in zip(axes, (int(r["i"]), int(r["j"]))):
            row = df.iloc[k]
            with Image.open(row["path"]) as im:
                ax.imshow(_display(im))
            ax.set_title(f"{row['source']} | grau {row['label']}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"distância = {int(r['dist'])}", fontsize=10)
        plt.show()


def stratified_split(df):
    key = lambda d: d["source"] + "_" + d["label"].astype(str)
    train, rest = train_test_split(df, train_size=SPLITS[0], stratify=key(df), random_state=SEED)
    val_frac = SPLITS[1] / (SPLITS[1] + SPLITS[2])
    val, test = train_test_split(rest, train_size=val_frac, stratify=key(rest), random_state=SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
