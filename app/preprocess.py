"""
Pré-processamento determinístico compartilhado entre o treino (Colab) e a inferência (Raspberry Pi).

Etapas: localizar a retina -> recortar -> completar com preto até virar quadrado -> redimensionar.

A normalização de cada modelo NÃO fica aqui: ela é a primeira camada do próprio modelo,
então o .tflite recebe a imagem em uint8 exatamente como sai deste módulo.

Depende só de NumPy e Pillow, para rodar igual no Colab e no Pi.
"""
import numpy as np
from PIL import Image

# LANCZOS: melhor filtro para reduzir imagens. Fallback para Pillow < 9.1.
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

BRIGHTNESS_TOL = 10         # pixel (em escala de cinza) acima disso conta como retina
MIN_LINE_FRACTION = 0.01    # linha/coluna só conta se >= 1% dos pixels forem retina (ignora ruído e pixels soltos)
MIN_RETINA_FRACTION = 0.10  # caixa menor que 10% da imagem = retina não encontrada


def find_retina_bbox(im, tol=BRIGHTNESS_TOL):
    """Retorna a caixa (esquerda, topo, direita, base) que contém a retina, ou None se não encontrar."""
    mask = np.asarray(im.convert("L")) > tol
    h, w = mask.shape
    rows = np.nonzero(mask.sum(axis=1) > MIN_LINE_FRACTION * w)[0]
    cols = np.nonzero(mask.sum(axis=0) > MIN_LINE_FRACTION * h)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    if (box[2] - box[0]) * (box[3] - box[1]) < MIN_RETINA_FRACTION * w * h:
        return None
    return box


def pad_to_square(im, fill=(0, 0, 0)):
    """Centraliza a imagem num quadrado preto. Evita esticar a retina quando o círculo vem cortado."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess_image(im, size, tol=BRIGHTNESS_TOL):
    """Imagem PIL -> (imagem RGB size x size, retina_encontrada).

    Se a retina não for encontrada, a imagem inteira é usada e retina_encontrada = False.
    Quem chama decide o que fazer: no treino, registrar; no Pi, avisar o usuário em vez de classificar.
    """
    im = im.convert("RGB")
    box = find_retina_bbox(im, tol)
    if box is not None:
        im = im.crop(box)
    return pad_to_square(im).resize((size, size), RESAMPLE), box is not None


def load_and_preprocess(path, size):
    """Atalho: abre o arquivo e aplica preprocess_image."""
    with Image.open(path) as im:
        return preprocess_image(im, size)


def to_array(im):
    """PIL -> np.uint8 (altura, largura, 3), o formato de entrada do modelo e do .tflite."""
    return np.asarray(im, dtype=np.uint8)
