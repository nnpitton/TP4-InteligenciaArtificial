"""
TP4 - Inteligencia Artificial - Pitton Juan Nicolas
Transformada de Hough aplicada a la foto del motor (block) del caso.

Uso:
    python tp4_hough.py                   -> corre los dos prototipos con motor_canvas.jpg
    python tp4_hough.py rectas            -> solo el prototipo de rectas (punto 2)
    python tp4_hough.py circulos          -> solo el prototipo de circunferencias (punto 3)
    python tp4_hough.py todo otra_foto.jpg salida

Imprime la salida por bloques y guarda en la carpeta de salida la foto de
entrada y las fotos procesadas con las rectas y circunferencias detectadas.
Todo el procesamiento de imagen (bordes y transformadas) esta en hough.py;
PIL se usa solo para leer la foto y matplotlib para guardar las figuras.
"""
import sys, os, time, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hough import (to_gray, canny, gaussian_blur, hough_lines, hough_lines_mb, hough_circle_fixed_r,
                   hough_circle_radius_range, find_peaks, refine_peak)

MODO = sys.argv[1] if len(sys.argv) > 1 else "todo"
FOTO = sys.argv[2] if len(sys.argv) > 2 else "motor_canvas.jpg"
OUT = sys.argv[3] if len(sys.argv) > 3 else "salida"
os.makedirs(OUT, exist_ok=True)
RES, LOG = {}, {}

# ---------------------------------------------------------------- parametros
ANCHO_UTIL = 440            # la foto de Canvas trae un recuadro ampliado a la derecha: se usa la vista principal
SAT_ANOTACION = 0.35        # saturacion minima para considerar un pixel como dibujo de color (ejes, flechas)
CANNY_RECTAS = dict(sigma=1.6, low=0.10, high=0.25)   # mas suavizado: las aristas largas lo toleran
CANNY_CIRC = dict(sigma=1.2, low=0.10, high=0.25)     # menos suavizado: conserva el contorno del aro
SUAVIZADO_ACC = 2.0         # el aro se ve como elipse: se suaviza el acumulador para juntar los votos
# rectas
PASO_THETA, PASO_RHO, N_RECTAS, VOTOS_MIN_RECTA = 0.5, 1.0, 10, 40
# circunferencias: radios medidos sobre la foto (el aro se ve como elipse por la perspectiva)
R_EXT = range(24, 29)       # borde exterior del aro C: 24 a 28 px
R_INT = range(15, 20)       # borde interior del aro C: 15 a 19 px
A_NOMINAL = (300, 200)      # posicion de A que el robot tiene programada (px)
VENTANA = 45                # el desplazamiento esperado es chico: se busca A a +-45 px de la nominal
R_AGUJEROS = np.arange(5, 10)   # agujeros de bulones, para mostrar las demas circunferencias


def bloque(clave, lineas):
    texto = "\n".join(lineas)
    print(texto + "\n")
    LOG[clave] = texto


def ms(t0):
    return 1e3 * (time.perf_counter() - t0)


# ---------------------------------------------------------------- utilidades de imagen (propias)
def dilatar(m, k):
    for _ in range(k):
        p = np.pad(m, 1)
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def limpiar_anotaciones(rgb):
    """
    La foto del enunciado tiene dibujados ejes, flechas y lineas de colores.
    El block es gris (saturacion baja), asi que los pixeles saturados son dibujos:
    se marcan, se rellenan con el promedio de sus vecinos y sus bordes se descartan.
    """
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    mask = dilatar((sat > SAT_ANOTACION) & (mx > 0.25), 2)
    g = to_gray(rgb)
    f, known = g.copy(), ~mask
    for _ in range(80):
        p = np.pad(f * known, 1); k = np.pad(known.astype(float), 1)
        s = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        c = k[:-2, 1:-1] + k[2:, 1:-1] + k[1:-1, :-2] + k[1:-1, 2:]
        nuevo = (~known) & (c > 0)
        f[nuevo] = s[nuevo] / c[nuevo]; known = known | nuevo
        if known.all(): break
    return f, mask


def bordes(rgb, params):
    g, mask = limpiar_anotaciones(rgb)
    e, gx, gy = canny(g, **params)
    e &= ~dilatar(mask, 3)
    return g, mask, e, gx, gy


def mover(rgb, ang_deg=0.0, dx=0.0, dy=0.0):
    """Gira la foto ang_deg alrededor de su centro y la desplaza (dx, dy), con
    interpolacion bilineal y fondo blanco. Simula el block movido sobre la cinta."""
    H, W, _ = rgb.shape
    cx, cy = W / 2, H / 2
    a = np.deg2rad(ang_deg)
    y, x = np.mgrid[0:H, 0:W].astype(float)
    # transformacion inversa: de la imagen de salida a la de entrada
    xr, yr = x - cx - dx, y - cy - dy
    xs = np.cos(a) * xr + np.sin(a) * yr + cx
    ys = -np.sin(a) * xr + np.cos(a) * yr + cy
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = (xs - x0)[..., None], (ys - y0)[..., None]
    out = np.ones_like(rgb)
    ok = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    x0c, y0c = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
    v = (rgb[y0c, x0c] * (1 - fx) * (1 - fy) + rgb[y0c, x0c + 1] * fx * (1 - fy)
         + rgb[y0c + 1, x0c] * (1 - fx) * fy + rgb[y0c + 1, x0c + 1] * fx * fy)
    out[ok] = v[ok]
    return out


def con_ruido(rgb, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rgb + rng.normal(0, sigma, rgb.shape[:2])[..., None], 0, 1)


# ---------------------------------------------------------------- prototipo 1: rectas
def detectar_rectas(rgb):
    g, mask, e, gx, gy = bordes(rgb, CANNY_RECTAS)
    acc, th, rh = hough_lines(e, PASO_THETA, PASO_RHO)
    pk = find_peaks(acc, N_RECTAS, VOTOS_MIN_RECTA, (12, 10))
    rectas = [(float(np.rad2deg(th[i[1]])), float(rh[i[0]]), int(v)) for i, v in pk]
    return rectas, acc, th, rh, e, mask


def dibujar_rectas(ax, rectas, W, H, color="lime"):
    for t, r, v in rectas:
        tt = np.deg2rad(t)
        if abs(np.sin(tt)) > 0.5:
            xs = np.array([0, W]); ys = (r - xs * np.cos(tt)) / np.sin(tt)
        else:
            ys = np.array([0, H]); xs = (r - ys * np.sin(tt)) / np.cos(tt)
        ax.plot(xs, ys, "-", color=color, lw=1.4)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


def tramos(e, t_deg, rho, tol=1.5, hueco=8, minimo=25):
    """Tramos de la recta (theta, rho) que tienen bordes encima: se proyectan los
    pixeles de borde cercanos sobre la recta y se agrupan si estan a menos de 'hueco' px."""
    ys, xs = np.nonzero(e)
    t = np.deg2rad(t_deg)
    d = xs * np.cos(t) + ys * np.sin(t) - rho
    sel = np.abs(d) <= tol
    s = np.sort(-xs[sel] * np.sin(t) + ys[sel] * np.cos(t))   # posicion a lo largo de la recta
    out = []
    if len(s) == 0: return out
    ini = s[0]
    for a_, b_ in zip(s[:-1], s[1:]):
        if b_ - a_ > hueco:
            if a_ - ini >= minimo: out.append((ini, a_))
            ini = b_
    if s[-1] - ini >= minimo: out.append((ini, s[-1]))
    pts = []
    for u0, u1 in out:   # punto = rho*n + u*direccion
        p0 = (rho * np.cos(t) - u0 * np.sin(t), rho * np.sin(t) + u0 * np.cos(t))
        p1 = (rho * np.cos(t) - u1 * np.sin(t), rho * np.sin(t) + u1 * np.cos(t))
        pts.append((p0, p1))
    return pts


def prototipo_rectas(rgb):
    H, W, _ = rgb.shape
    t0 = time.perf_counter()
    rectas, acc, th, rh, e, mask = detectar_rectas(rgb)
    t_tot = ms(t0)
    bloque("r1", ["=== Punto 2: Hough de rectas sobre la foto del motor ===",
                  f"Foto: {FOTO} | zona usada: {W} x {H} px",
                  f"Pixeles de anotaciones de color descartados: {int(mask.sum())} ({100*mask.mean():.1f} %)",
                  f"Pixeles de borde (Canny propio): {int(e.sum())}",
                  f"Acumulador (theta, rho): {len(th)} x {len(rh)} celdas | paso {PASO_THETA} grados y {PASO_RHO:.0f} px",
                  f"Tiempo total: {t_tot:.0f} ms"])
    lin = [f"Rectas detectadas (las {len(rectas)} con mas votos):", "  #   theta(grados)   rho(px)   votos"]
    for k, (t, r, v) in enumerate(rectas, 1):
        lin.append(f"  {k:<3} {t:8.1f}      {r:8.1f}   {v:5d}")
    bloque("r2", lin)

    # salida: foto original y foto con las rectas
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9))
    ax[0].imshow(rgb); ax[0].set_title("Entrada: foto del motor")
    ax[1].imshow(rgb); dibujar_rectas(ax[1], rectas, W, H, color="#7CFC00")
    for ln in ax[1].lines: ln.set_alpha(0.35); ln.set_linewidth(0.9)
    n_tr = 0
    for t, r_, v in rectas:
        for (x0, y0), (x1, y1) in tramos(e, t, r_):
            ax[1].plot([x0, x1], [y0, y1], "-", color="red", lw=2.6); n_tr += 1
    ax[1].set_title(f"Salida: {len(rectas)} rectas (tramos sobre aristas en rojo)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/rectas_entrada_salida.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].imshow(e, cmap="gray"); ax[0].set_title(f"Bordes sin anotaciones: {e.sum()} píxeles"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(np.sqrt(acc), cmap="magma", aspect="auto", extent=[-90, 90, rh[-1], rh[0]])
    for t, r, v in rectas: ax[1].plot(t, r, "o", mfc="none", mec="cyan", ms=9)
    ax[1].set_xlabel("θ (grados)"); ax[1].set_ylabel("ρ (píxeles)"); ax[1].set_title("Acumulador (θ, ρ) y picos")
    fig.tight_layout(); fig.savefig(f"{OUT}/rectas_bordes_acumulador.png", dpi=150); plt.close(fig)

    # ensayo: giro sobre Z conocido
    ref = rectas[:5]
    lin = ["Ensayo de giro: se gira la foto un angulo conocido y se mide el giro con las 5 rectas principales",
           "  giro real   giro medido (mediana)   rectas que coinciden"]
    giros = []
    for ang in (-6.0, -3.0, 2.0, 5.0):
        rr, *_ = detectar_rectas(mover(rgb, ang))
        med = []
        for t, r, v in ref:
            cand = [x for x in rr if abs(((x[0] - t - ang) + 90) % 180 - 90) <= 1.5]
            if cand:
                b = max(cand, key=lambda x: x[2])
                med.append(((b[0] - t) + 90) % 180 - 90)
        m = float(np.median(med)) if med else float("nan")
        giros.append(dict(real=ang, medido=m, coinciden=len(med)))
        lin.append(f"  {ang:+6.1f}        {m:+6.2f}                  {len(med)} de 5")
    bloque("r3", lin)

    # ensayo: ruido sobre la foto real
    lin = ["Ensayo de ruido: ruido gaussiano sumado a la foto",
           "  ruido   pixeles borde   rectas principales encontradas (de 5)"]
    ruido = []
    for s in (0.0, 0.05, 0.10, 0.15, 0.20):
        rr, _, _, _, e2, _ = detectar_rectas(con_ruido(rgb, s, 1))
        ok = sum(1 for t, r, v in ref if any(abs(x[0] - t) <= 1.0 and abs(x[1] - r) <= 3 for x in rr))
        ruido.append(dict(sigma=s, bordes=int(e2.sum()), ok=ok))
        lin.append(f"  {s:4.2f}    {int(e2.sum()):8d}         {ok}")
    bloque("r4", lin)
    # ensayo: que pasa si no se limpian las anotaciones de la foto
    g0 = to_gray(rgb); e0, _, _ = canny(g0, **CANNY_RECTAS)
    a0, th0, rh0 = hough_lines(e0, PASO_THETA, PASO_RHO)
    ys0, xs0 = np.nonzero(e0)
    lin = ["Sin limpiar las anotaciones de color (ejes y flechas dibujados en la foto):",
           "  #   theta(grados)   rho(px)   votos   bordes sobre anotaciones"]
    sucias = 0
    for k, (i, v) in enumerate(find_peaks(a0, N_RECTAS, VOTOS_MIN_RECTA, (12, 10)), 1):
        t = th0[i[1]]; rr_ = rh0[i[0]]
        sel = np.abs(xs0 * np.cos(t) + ys0 * np.sin(t) - rr_) <= 1.5
        fr = float(mask[ys0[sel], xs0[sel]].mean())
        sucias += fr > 0.5
        lin.append(f"  {k:<3} {np.rad2deg(t):8.1f}      {rr_:8.1f}   {v:5d}     {100*fr:3.0f} %")
    lin.append(f"Rectas que caen mayormente sobre anotaciones: {sucias} de {N_RECTAS}")
    bloque("r5", lin)

    RES["rectas"] = dict(sucias=int(sucias), n_mask=int(mask.sum()), pct_mask=float(100 * mask.mean()), n_bordes=int(e.sum()),
                         t_ms=t_tot, rectas=rectas, giros=giros, ruido=ruido)


# ---------------------------------------------------------------- prototipo 2: circunferencias
def localizar_A(rgb, modo="polaridad", r_ext=R_EXT, r_int=R_INT):
    """
    modo "polaridad": el aro es claro sobre fondo oscuro. En el borde exterior el
      centro esta del lado claro (-R) y en el interior del lado oscuro (+R).
    modo "gradiente": vota a +R y -R (sin usar la polaridad).
    modo "completo": cada borde vota por toda la circunferencia.
    """
    g, mask, e, gx, gy = bordes(rgb, CANNY_CIRC)
    t0 = time.perf_counter()
    acc = np.zeros(e.shape)
    for R in r_ext:       # banda de radios: todos votan al mismo centro
        acc += (hough_circle_fixed_r(e, R) if modo == "completo" else
                hough_circle_fixed_r(e, R, gx, gy, spread_deg=3, sentido=-1 if modo == "polaridad" else None))
    for R in r_int:
        acc += (hough_circle_fixed_r(e, R) if modo == "completo" else
                hough_circle_fixed_r(e, R, gx, gy, spread_deg=3, sentido=+1 if modo == "polaridad" else None))
    if SUAVIZADO_ACC > 0:
        acc = gaussian_blur(acc.astype(float), SUAVIZADO_ACC)
    th = ms(t0)
    x0, y0 = A_NOMINAL
    win = np.zeros_like(acc); sl = (slice(y0 - VENTANA, y0 + VENTANA + 1), slice(x0 - VENTANA, x0 + VENTANA + 1))
    win[sl] = acc[sl]
    pk = find_peaks(win, 2, 0, 8)
    (iy, ix), v = pk[0]
    yA, xA = refine_peak(acc, (iy, ix), r=3)
    nit = pk[0][1] / max(pk[1][1], 1) if len(pk) > 1 else float("inf")
    return dict(xA=float(xA), yA=float(yA), votos=float(v), nitidez=float(nit), t_hough=th,
                acc=acc, e=e, gx=gx, gy=gy, mask=mask)


def prototipo_circulos(rgb):
    H, W, _ = rgb.shape
    t0 = time.perf_counter()
    r = localizar_A(rgb)
    t_tot = ms(t0)
    # demas circunferencias de la foto (agujeros de bulones), con radio variable
    a3 = hough_circle_radius_range(r["e"], R_AGUJEROS, r["gx"], r["gy"], spread_deg=3)
    otros = []
    for (ir, iy, ix), v in find_peaks(a3, 25, 0.9, (3, 7, 7)):
        if np.hypot(ix - r["xA"], iy - r["yA"]) > 30:
            otros.append((int(ix), int(iy), int(R_AGUJEROS[ir]), float(v)))
    otros = otros[:12]
    bloque("c1", ["=== Punto 3: Hough de circunferencias sobre la foto del motor ===",
                  f"Foto: {FOTO} | zona usada: {W} x {H} px",
                  f"Aro C (medido en la foto): borde exterior {R_EXT.start}-{R_EXT.stop-1} px, interior {R_INT.start}-{R_INT.stop-1} px",
                  f"Posicion nominal de A (programada en el robot): X = {A_NOMINAL[0]}, Y = {A_NOMINAL[1]} px",
                  f"Ventana de busqueda: +-{VENTANA} px | pixeles de borde: {int(r['e'].sum())}"])
    bloque("c2", ["Resultado:",
                  f"  Centro A detectado: X = {r['xA']:.2f}, Y = {r['yA']:.2f} px",
                  f"  Votos en el pico (acumulador suavizado): {r['votos']:.1f} | nitidez (pico 1 / pico 2): {r['nitidez']:.2f}",
                  f"  Desvio respecto de la nominal: dX = {r['xA']-A_NOMINAL[0]:+.2f} px, dY = {r['yA']-A_NOMINAL[1]:+.2f} px",
                  f"  Otras circunferencias encontradas (agujeros de bulones): {len(otros)}",
                  f"  Tiempo total: {t_tot:.0f} ms (votacion {r['t_hough']:.0f} ms)"])

    # salida: foto original y foto con las circunferencias
    tt = np.linspace(0, 2 * np.pi, 120)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9))
    ax[0].imshow(rgb); ax[0].set_title("Entrada: foto del motor")
    ax[1].imshow(rgb)
    for ix, iy, R, v in otros:
        ax[1].plot(ix + R * np.cos(tt), iy + R * np.sin(tt), "-", color="cyan", lw=1.2)
    for R in (R_EXT.start + 2, R_INT.start + 2):
        ax[1].plot(r["xA"] + R * np.cos(tt), r["yA"] + R * np.sin(tt), "-", color="lime", lw=1.8)
    ax[1].plot(r["xA"], r["yA"], "+", color="red", ms=14, mew=2.2)
    ax[1].set_title("Salida: aro C y centro A (verde/rojo), agujeros (celeste)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/circ_entrada_salida.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    ax[0].imshow(r["e"], cmap="gray"); ax[0].set_title("Bordes sin anotaciones")
    ax[1].imshow(np.sqrt(r["acc"]), cmap="magma"); ax[1].set_title("Acumulador (x₀, y₀)")
    x0, y0 = A_NOMINAL
    ax[1].add_patch(plt.Rectangle((x0 - VENTANA, y0 - VENTANA), 2 * VENTANA, 2 * VENTANA, fill=False, ec="cyan", lw=1))
    z = (slice(int(r["yA"]) - 45, int(r["yA"]) + 46), slice(int(r["xA"]) - 45, int(r["xA"]) + 46))
    ax[2].imshow(rgb[z]); ax[2].plot(45 + r["xA"] % 1, 45 + r["yA"] % 1, "+", color="red", ms=16, mew=2.2)
    for R in (R_EXT.start + 2, R_INT.start + 2):
        ax[2].plot(45 + R * np.cos(tt), 45 + R * np.sin(tt), "-", color="lime", lw=1.5)
    ax[2].set_title("Detalle del aro C y el centro A")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/circ_bordes_acumulador.png", dpi=150); plt.close(fig)

    # comparacion de formas de votar
    rg = localizar_A(rgb, "gradiente")
    rc = localizar_A(rgb, "completo")
    r1 = localizar_A(rgb, "polaridad", R_EXT, [])
    rf = localizar_A(rgb, "polaridad", [26], [17])
    lin = ["Formas de votar (sobre la misma foto):",
           "  variante                          X A      Y A     votos   nitidez   votacion"]
    var = {}
    for nombre, x in (("polaridad, banda ext + int", r), ("gradiente +-R, banda ext + int", rg),
                      ("voto completo, banda ext + int", rc), ("polaridad, solo banda ext", r1),
                      ("polaridad, radio fijo 26 + 17", rf)):
        var[nombre] = {k: x[k] for k in ("xA", "yA", "votos", "nitidez", "t_hough")}
        lin.append(f"  {nombre:<32} {x['xA']:7.2f}  {x['yA']:7.2f}  {x['votos']:6.1f}    {x['nitidez']:5.2f}   {x['t_hough']:6.0f} ms")
    bloque("c3", lin)

    # ensayo: desplazamiento conocido del block
    lin = ["Ensayo de desplazamiento: se mueve la foto una distancia conocida y se mide cuanto se movio A",
           "  desplazamiento real (px)   medido (px)        error (px)"]
    desp = []
    for dx, dy in ((12.3, -8.6), (-15.4, 6.2), (7.5, 10.3), (-20.7, -12.2), (24.6, 15.4), (-4.2, -3.7)):
        q = localizar_A(mover(rgb, 0, dx, dy))
        mx_, my_ = q["xA"] - r["xA"], q["yA"] - r["yA"]
        err = float(np.hypot(mx_ - dx, my_ - dy))
        desp.append(dict(dx=dx, dy=dy, mx=mx_, my=my_, err=err))
        lin.append(f"  ({dx:+6.1f}, {dy:+6.1f})           ({mx_:+6.2f}, {my_:+6.2f})   {err:5.2f}")
    lin.append(f"  Error medio: {np.mean([d['err'] for d in desp]):.2f} px | maximo: {max(d['err'] for d in desp):.2f} px")
    bloque("c4", lin)

    # ensayo: ruido
    lin = ["Ensayo de ruido: ruido gaussiano sumado a la foto", "  ruido   X A      Y A      error respecto de la foto limpia (px)"]
    ruido = []
    for s in (0.05, 0.10, 0.15, 0.20):
        q = localizar_A(con_ruido(rgb, s, 2))
        err = float(np.hypot(q["xA"] - r["xA"], q["yA"] - r["yA"]))
        ruido.append(dict(sigma=s, err=err))
        lin.append(f"  {s:4.2f}   {q['xA']:7.2f}  {q['yA']:7.2f}   {err:5.2f}")
    bloque("c5", lin)
    # ensayo: aporte de cada decision (suavizado del acumulador y polaridad)
    global SUAVIZADO_ACC
    sm_orig = SUAVIZADO_ACC
    lin = ["Aporte de cada decision (mismos ensayos de desplazamiento y ruido):",
           "  suavizado  voto           error medio desp.  error max desp.  error max ruido"]
    abl = []
    shifts = [(d["dx"], d["dy"]) for d in desp]
    for sm in (0.0, sm_orig):
        SUAVIZADO_ACC = sm
        for modo in ("gradiente", "polaridad"):
            b0 = localizar_A(rgb, modo)
            dd = [np.hypot(q["xA"] - b0["xA"] - dx, q["yA"] - b0["yA"] - dy)
                  for dx, dy in shifts for q in [localizar_A(mover(rgb, 0, dx, dy), modo)]]
            nn = [np.hypot(q["xA"] - b0["xA"], q["yA"] - b0["yA"])
                  for s_ in (0.05, 0.10, 0.15, 0.20) for q in [localizar_A(con_ruido(rgb, s_, 2), modo)]]
            abl.append(dict(suavizado=sm, modo=modo, medio=float(np.mean(dd)), maximo=float(np.max(dd)), ruido=float(np.max(nn))))
            lin.append(f"  {'si' if sm else 'no':<9}  {modo:<13}  {np.mean(dd):8.2f}          {np.max(dd):8.2f}         {np.max(nn):8.2f}")
    SUAVIZADO_ACC = sm_orig
    bloque("c6", lin)

    RES["circ"] = dict(ablacion=abl, xA=r["xA"], yA=r["yA"], votos=r["votos"], nitidez=r["nitidez"], t_ms=t_tot,
                       n_bordes=int(r["e"].sum()), otros=otros, variantes=var, desplazamientos=desp, ruido=ruido)


# ---------------------------------------------------------------- ejemplo conceptual (punto 1)
def ejemplo_mb():
    sq = np.zeros((120, 160), dtype=bool)
    sq[20:100, 50] = True
    for k in range(140):
        yy = int(round(10 + 0.5 * k))
        if yy < 120: sq[yy, 10 + k] = True
    accmb, msl, bs = hough_lines_mb(sq)
    acct, th2, rh2 = hough_lines(sq)
    pk_mb = find_peaks(accmb, 2, 20, (10, 10)); pk_t = find_peaks(acct, 2, 20, (10, 10))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(sq, cmap="gray_r"); ax[0].set_title("a) Arista vertical y oblicua"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(np.sqrt(accmb), cmap="magma", aspect="auto", extent=[msl[0], msl[-1], bs[-1], bs[0]])
    ax[1].set_ylim(150, -150); ax[1].set_xlabel("m"); ax[1].set_ylabel("b"); ax[1].set_title("b) Plano (m, b), m ∈ [-5, 5]")
    for i, v in pk_mb: ax[1].plot(msl[i[1]], bs[i[0]], "o", mfc="none", mec="cyan", ms=10)
    ax[2].imshow(np.sqrt(acct), cmap="magma", aspect="auto", extent=[-90, 90, rh2[-1], rh2[0]])
    ax[2].set_xlabel("θ (grados)"); ax[2].set_ylabel("ρ"); ax[2].set_title("c) Plano (θ, ρ)")
    for i, v in pk_t: ax[2].plot(np.rad2deg(th2[i[1]]), rh2[i[0]], "o", mfc="none", mec="cyan", ms=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/ejemplo_mb_theta_rho.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- programa principal
rgb = np.asarray(Image.open(FOTO).convert("RGB"), dtype=np.float64) / 255.0
rgb = rgb[:, :ANCHO_UTIL]
Image.fromarray((rgb * 255).astype(np.uint8)).save(f"{OUT}/foto_entrada.png")
if MODO in ("rectas", "todo"):
    prototipo_rectas(rgb)
if MODO in ("circulos", "todo"):
    prototipo_circulos(rgb)
if MODO == "todo":
    ejemplo_mb()
with open(os.path.join(OUT, "resultados.json"), "w", encoding="utf-8") as fh:
    json.dump(RES, fh, indent=1, default=float)
with open(os.path.join(OUT, "consola.json"), "w", encoding="utf-8") as fh:
    json.dump(LOG, fh, indent=1, ensure_ascii=False)
