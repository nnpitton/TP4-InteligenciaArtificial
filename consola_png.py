"""
Convierte cada bloque de salida en una imagen tipo consola (fondo oscuro).
Uso: python consola_png.py salida/consola.json salida
Funciona en Windows, macOS y Linux: busca una fuente monoespaciada instalada
y, si no encuentra ninguna, usa la fuente que trae Pillow.
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("salida", "consola.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(src) or "."
TAM = 17

CANDIDATAS = [
    # Windows
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "consola.ttf"),   # Consolas
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "CascadiaMono.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "lucon.ttf"),     # Lucida Console
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "cour.ttf"),      # Courier New
    # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def cargar_fuente():
    for ruta in CANDIDATAS:
        if os.path.exists(ruta):
            try:
                return ImageFont.truetype(ruta, TAM)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=TAM)     # Pillow 10.1 o superior
    except TypeError:
        return ImageFont.load_default()


font = cargar_fuente()
with open(src, encoding="utf-8") as fh:
    log = json.load(fh)
os.makedirs(out, exist_ok=True)

for k, txt in log.items():
    lines = txt.split("\n")
    ancho = max(font.getlength(l) for l in lines)
    asc, desc = font.getmetrics()
    lh = asc + desc + 4
    im = Image.new("RGB", (int(ancho) + 22, lh * len(lines) + 16), (12, 12, 12))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((11, 8 + i * lh), l, font=font, fill=(204, 204, 204))
    ruta = os.path.join(out, f"cons_{k}.png")
    im.save(ruta)
    print("guardada", ruta)
