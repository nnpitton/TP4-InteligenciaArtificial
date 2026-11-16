"""
TP4 - Transformada de Hough (rectas y circunferencias)
Implementación propia sobre NumPy: no se utiliza ninguna librería de visión
(OpenCV, scikit-image, etc.). NumPy se usa solo como soporte de arreglos;
PIL/matplotlib solo para leer/guardar imágenes y graficar.

Contenido:
  - Preprocesamiento: escala de grises, suavizado gaussiano, gradiente de Sobel,
    detector de bordes tipo Canny (supresión de no máximos + doble umbral).
  - Hough de rectas en el espacio (theta, rho).
  - Hough de circunferencias de radio conocido (espacio 2D: xo, yo),
    con voto completo o voto orientado por gradiente.
  - Hough de circunferencias de radio desconocido (espacio 3D: xo, yo, R).
  - Búsqueda de picos con supresión de vecindario y refinamiento subpíxel.
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Preprocesamiento
# ---------------------------------------------------------------------------
def to_gray(img):
    """RGB (H,W,3) uint8 -> gris float en [0,1] (luminancia ITU-R BT.601)."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _convolve_1d(img, k, axis):
    """Convolución 1D con relleno por reflexión a lo largo de un eje."""
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img)
    n = img.shape[axis]
    for i, w in enumerate(k):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        out += w * p[tuple(sl)]
    return out


def gaussian_blur(img, sigma=1.4):
    """Filtro gaussiano separable (dos convoluciones 1D)."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return _convolve_1d(_convolve_1d(img, k, 0), k, 1)


def sobel(img):
    """Gradientes gx, gy con el operador de Sobel (separable)."""
    d = np.array([1.0, 0.0, -1.0])   # derivada
    s = np.array([1.0, 2.0, 1.0])    # suavizado
    gx = _convolve_1d(_convolve_1d(img, s, 0), d, 1)
    gy = _convolve_1d(_convolve_1d(img, d, 0), s, 1)
    return gx, gy


def canny(img, sigma=1.4, low=0.1, high=0.25):
    """
    Detector de bordes tipo Canny implementado paso a paso.
    low/high: umbrales relativos al máximo de la magnitud del gradiente.
    Devuelve: mapa binario de bordes, gx, gy.
    """
    g = gaussian_blur(img, sigma)
    gx, gy = sobel(g)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

    # Supresión de no máximos: se conserva el píxel solo si es máximo
    # local en la dirección del gradiente (4 direcciones cuantizadas).
    H, W = mag.shape
    m = np.pad(mag, 1)
    c = m[1:-1, 1:-1]
    nms = np.zeros_like(mag)
    dirs = [((ang < 22.5) | (ang >= 157.5), (0, 1)),
            ((ang >= 22.5) & (ang < 67.5), (1, 1)),
            ((ang >= 67.5) & (ang < 112.5), (1, 0)),
            ((ang >= 112.5) & (ang < 157.5), (1, -1))]
    for mask, (dy, dx) in dirs:
        a = m[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
        b = m[1 - dy:H + 1 - dy, 1 - dx:W + 1 - dx]
        keep = mask & (c >= a) & (c >= b)
        nms[keep] = c[keep]

    # Doble umbral + histéresis (propagación de bordes fuertes a débiles).
    hi = high * nms.max()
    lo = low * nms.max()
    strong = nms >= hi
    weak = (nms >= lo) & ~strong
    edges = strong.copy()
    while True:
        p = np.pad(edges, 1)
        neigh = np.zeros_like(edges)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neigh |= p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
        new = edges | (weak & neigh)
        if new.sum() == edges.sum():
            break
        edges = new
    return edges, gx, gy


# ---------------------------------------------------------------------------
# 2. Búsqueda de picos en un acumulador
# ---------------------------------------------------------------------------
def find_peaks(acc, n, min_votes, nbhd):
    """
    Devuelve hasta n máximos del acumulador (índices) con al menos min_votes
    votos. Tras elegir un pico se anula su vecindario (tamaño nbhd por eje)
    para no volver a elegir el mismo objeto.
    """
    a = acc.astype(np.float64).copy()
    peaks = []
    nbhd = np.atleast_1d(nbhd)
    if len(nbhd) == 1:
        nbhd = np.repeat(nbhd, a.ndim)
    for _ in range(n):
        idx = np.unravel_index(np.argmax(a), a.shape)
        if a[idx] < min_votes:
            break
        peaks.append((idx, acc[idx]))
        sl = tuple(slice(max(0, i - r), i + r + 1) for i, r in zip(idx, nbhd))
        a[sl] = 0
    return peaks


def refine_peak(acc, idx, r=2):
    """Centroide ponderado alrededor del pico -> posición subpíxel."""
    sl = tuple(slice(max(0, i - r), i + r + 1) for i in idx)
    w = acc[sl].astype(np.float64)
    grids = np.meshgrid(*[np.arange(s.start, s.start + w.shape[k])
                          for k, s in enumerate(sl)], indexing="ij")
    tot = w.sum()
    return tuple((g * w).sum() / tot for g in grids)


# ---------------------------------------------------------------------------
# 3. Hough de rectas: rho = x cos(theta) + y sin(theta)
# ---------------------------------------------------------------------------
def hough_lines(edges, theta_step_deg=0.5, rho_step=1.0):
    """
    Cada píxel de borde (x, y) vota por todas las rectas que pasan por él:
    en el plano (theta, rho) traza una sinusoide.
    Devuelve acumulador [n_rho, n_theta], vector theta (rad) y vector rho.
    """
    H, W = edges.shape
    diag = np.hypot(H, W)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step_deg))
    rhos = np.arange(-diag, diag + rho_step, rho_step)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    ys, xs = np.nonzero(edges)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    t_idx = np.arange(len(thetas))
    # Procesar por bloques para acotar memoria.
    for k in range(0, len(xs), 4000):
        x = xs[k:k + 4000, None]
        y = ys[k:k + 4000, None]
        rho = x * cos_t + y * sin_t                       # (n, T)
        r_idx = np.round((rho + diag) / rho_step).astype(np.int64)
        np.add.at(acc, (r_idx.ravel(), np.broadcast_to(t_idx, r_idx.shape).ravel()), 1)
    return acc, thetas, rhos


def hough_lines_mb(edges, m_range=(-5, 5), m_steps=401, b_step=1.0):
    """
    Variante didáctica en el plano (m, b) con y = m x + b, para mostrar la
    limitación de las rectas verticales (m -> infinito).
    """
    H, W = edges.shape
    ms = np.linspace(m_range[0], m_range[1], m_steps)
    bmax = H + max(abs(m_range[0]), abs(m_range[1])) * W
    bs = np.arange(-bmax, bmax + b_step, b_step)
    acc = np.zeros((len(bs), len(ms)), dtype=np.int32)
    ys, xs = np.nonzero(edges)
    m_idx = np.arange(len(ms))
    for k in range(0, len(xs), 4000):
        b = ys[k:k + 4000, None] - ms * xs[k:k + 4000, None]
        b_idx = np.round((b + bmax) / b_step).astype(np.int64)
        np.add.at(acc, (b_idx.ravel(), np.broadcast_to(m_idx, b_idx.shape).ravel()), 1)
    return acc, ms, bs


# ---------------------------------------------------------------------------
# 4. Hough de circunferencias
# ---------------------------------------------------------------------------
def hough_circle_fixed_r(edges, R, gx=None, gy=None, n_angles=None, spread_deg=0, sentido=None):
    """
    Radio conocido -> espacio de parámetros 2D (xo, yo).
    Modo completo (gx=None): cada punto de borde vota por toda la
      circunferencia de radio R centrada en él (el "cono" cortado en R).
    Modo orientado (gx, gy dados): cada punto vota solo en la dirección
      del gradiente, a +R y -R (el centro está sobre la normal al borde);
      spread_deg permite tolerar error angular del gradiente.
    sentido (solo modo orientado): None vota a +R y -R; +1 o -1 vota de un solo
      lado. El gradiente de sobel() apunta hacia la zona mas oscura, asi que en un
      borde de un objeto claro el centro esta a -R (borde exterior de un aro claro)
      y en el borde de un agujero oscuro, a +R (borde interior).
    """
    H, W = edges.shape
    acc = np.zeros((H, W), dtype=np.int32)
    ys, xs = np.nonzero(edges)

    if gx is None:
        if n_angles is None:
            n_angles = max(90, int(2 * np.pi * R))   # ~1 voto por píxel de perímetro
        phi = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        dx, dy = R * np.cos(phi), R * np.sin(phi)
        for k in range(0, len(xs), 2000):
            xc = np.round(xs[k:k + 2000, None] - dx).astype(np.int64)
            yc = np.round(ys[k:k + 2000, None] - dy).astype(np.int64)
            ok = (xc >= 0) & (xc < W) & (yc >= 0) & (yc < H)
            np.add.at(acc, (yc[ok], xc[ok]), 1)
    else:
        phi0 = np.arctan2(gy[ys, xs], gx[ys, xs])[:, None]
        offs = np.deg2rad(np.arange(-spread_deg, spread_deg + 1, 1.0))[None, :]
        for sign in ((1, -1) if sentido is None else (sentido,)):
            phi = phi0 + offs
            xc = np.round(xs[:, None] + sign * R * np.cos(phi)).astype(np.int64)
            yc = np.round(ys[:, None] + sign * R * np.sin(phi)).astype(np.int64)
            ok = (xc >= 0) & (xc < W) & (yc >= 0) & (yc < H)
            np.add.at(acc, (yc[ok], xc[ok]), 1)
    return acc


def hough_circle_radius_range(edges, radii, gx, gy, spread_deg=2):
    """
    Radio desconocido -> espacio 3D (R, yo, xo). Se usa el voto orientado
    para que el costo sea manejable. Los votos se normalizan por el perímetro
    para no favorecer a las circunferencias grandes.
    """
    H, W = edges.shape
    acc = np.zeros((len(radii), H, W), dtype=np.float64)
    for i, R in enumerate(radii):
        acc[i] = hough_circle_fixed_r(edges, R, gx, gy, spread_deg=spread_deg) / (2 * np.pi * R)
    return acc
