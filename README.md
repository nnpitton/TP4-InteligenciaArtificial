# TP4 – Transformada de Hough

Dos prototipos de la transformada de Hough que trabajan sobre la **foto real del block del motor** del caso (`motor_canvas.jpg`, la figura 2 del enunciado en Canvas):

- **Rectas (punto 2):** detecta las aristas rectas del block y mide su giro sobre el eje Z.
- **Circunferencias (punto 3):** localiza el centro "A" del aro "C" y los agujeros de bulones.

Los dos prototipos están en un solo programa, que se ejecuta para uno u otro con un parámetro.

Todo el procesamiento de imágenes (escala de grises, filtro gaussiano, Sobel, detector de Canny y transformadas de Hough) está programado en `hough.py`. No se usa ninguna biblioteca de visión artificial. NumPy se usa solo para manejar arreglos, Pillow para leer la foto y matplotlib para guardar las figuras.

## Requisitos

- Python 3.9 o superior.
- Las librerías de `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Cómo ejecutarlo

Desde la carpeta `TP4` (la foto `motor_canvas.jpg` tiene que estar en la misma carpeta):

| Comando | Qué ejecuta |
| --- | --- |
| `python tp4_hough.py` | Los dos prototipos (igual que `python tp4_hough.py todo`). |
| `python tp4_hough.py rectas` | Solo el prototipo de rectas (punto 2). |
| `python tp4_hough.py circulos` | Solo el prototipo de circunferencias (punto 3). |
| `python tp4_hough.py todo otra_foto.jpg carpeta` | Con otra foto y otra carpeta de salida. |

En Windows, si `python` no se reconoce, usar `py` (por ejemplo, `py tp4_hough.py rectas`).

El programa completo tarda entre 15 y 30 segundos, porque además de procesar la foto repite el proceso para los ensayos (foto girada, desplazada y con ruido). Imprime la salida por bloques en la consola y guarda las imágenes en la carpeta `salida`.

Para guardar también la salida en un archivo de texto:

```bash
python tp4_hough.py > salida.txt
```

Opcional: para generar las imágenes tipo consola que se usaron en el informe:

```bash
python consola_png.py salida/consola.json salida
```

## Qué hace

1. **Lee la foto** y usa la vista principal del block (440 × 395 píxeles); descarta el recuadro ampliado de la derecha.
2. **Limpia las anotaciones.** La foto tiene dibujados ejes, flechas y una línea punteada de colores. Como el block es gris, los píxeles saturados se marcan como dibujo, se rellenan con sus vecinos y sus bordes se descartan.
3. **Detecta los bordes** con un detector de Canny propio: suavizado gaussiano, gradiente de Sobel, supresión de no máximos y doble umbral con histéresis.
4. **Rectas:**
   - vota en el espacio (θ, ρ), con ρ = x·cos θ + y·sen θ, y toma las 10 rectas con más votos;
   - dibuja en rojo los tramos de cada recta que se apoyan sobre aristas de la foto;
   - gira la foto ángulos conocidos para medir el giro sobre Z.
5. **Circunferencias:**
   - vota en el espacio (x₀, y₀) con una banda de radios medida en la foto (el aro se ve como elipse por la perspectiva), usando el gradiente y la polaridad del aro (claro sobre fondo oscuro);
   - suaviza el acumulador y busca el pico cerca de la posición nominal de A;
   - además busca los agujeros de bulones con radio variable;
   - desplaza la foto distancias conocidas para medir la precisión.

## Estructura del código

| Archivo | Funciones | Qué hace |
| --- | --- | --- |
| `hough.py` | `to_gray`, `gaussian_blur`, `sobel`, `canny` | Preprocesamiento y detección de bordes. |
| `hough.py` | `hough_lines`, `hough_lines_mb` | Hough de rectas en (θ, ρ) y, para comparar, en (m, b). |
| `hough.py` | `hough_circle_fixed_r`, `hough_circle_radius_range` | Hough de circunferencias con radio conocido (2D) y radio variable (3D). |
| `hough.py` | `find_peaks`, `refine_peak` | Búsqueda de picos con supresión de vecindario y centroide subpíxel. |
| `tp4_hough.py` | `limpiar_anotaciones`, `bordes`, `mover`, `con_ruido` | Limpieza de la foto, bordes, giro/desplazamiento de la foto y ruido para los ensayos. |
| `tp4_hough.py` | `prototipo_rectas`, `tramos` | Punto 2: rectas, salida y ensayos. |
| `tp4_hough.py` | `prototipo_circulos`, `localizar_A` | Punto 3: aro C, centro A, agujeros, salida y ensayos. |
| `consola_png.py` | — | Convierte cada bloque de la salida en una imagen tipo consola. |

Los parámetros (radios del aro, posición nominal de A, umbrales, suavizado) están al principio de `tp4_hough.py`. Si se usa otra foto, hay que ajustarlos.

## Salida

En la consola:

| Bloque | Contenido |
| --- | --- |
| Rectas 1 | Composición: foto, píxeles de anotaciones descartados, bordes, acumulador y tiempo. |
| Rectas 2 | Las 10 rectas detectadas (θ, ρ y votos). |
| Rectas 3 | Ensayo de giro: giro real contra giro medido. |
| Rectas 4 | Ensayo de ruido. |
| Rectas 5 | Qué pasa si no se limpian las anotaciones. |
| Circunferencias 1 | Composición: radios del aro, posición nominal y ventana de búsqueda. |
| Circunferencias 2 | Centro A detectado, votos y nitidez del pico. |
| Circunferencias 3 | Comparación de formas de votar. |
| Circunferencias 4 | Ensayo de desplazamiento: real contra medido. |
| Circunferencias 5 | Ensayo de ruido. |
| Circunferencias 6 | Aporte de cada decisión (suavizado del acumulador y polaridad). |

En la carpeta `salida`:

| Archivo | Contenido |
| --- | --- |
| `foto_entrada.png` | Foto de entrada (vista principal). |
| `rectas_entrada_salida.png` | Entrada y salida del prototipo de rectas. |
| `rectas_bordes_acumulador.png` | Bordes sin anotaciones y acumulador (θ, ρ). |
| `circ_entrada_salida.png` | Entrada y salida del prototipo de circunferencias. |
| `circ_bordes_acumulador.png` | Bordes, acumulador con la ventana de búsqueda y detalle del aro. |
| `ejemplo_mb_theta_rho.png` | Ejemplo conceptual: arista vertical en (m, b) y en (θ, ρ). |
| `resultados.json` | Todos los valores numéricos. |
| `consola.json` | Texto de cada bloque. |

## Resultados esperados

- **Rectas:** 10 rectas sobre aristas del block; giros conocidos medidos con error de 0 a 0,5°.
- **Circunferencias:** centro A en (299,03; 202,08) px; desplazamientos conocidos medidos con error medio de 0,82 px.

Los resultados son siempre los mismos (el ruido de los ensayos usa una semilla fija). Solo los tiempos en milisegundos cambian según la computadora.

## Limitaciones

- Se probó con una sola foto; en planta habría que validarlo con fotos de la línea.
- Por la perspectiva el aro se ve como una elipse; por eso se vota con una banda de radios.
- Los resultados están en píxeles; para pasarlos a milímetros hay que calibrar la cámara.
- Un círculo solo no permite medir el giro sobre Z; para eso se usa el prototipo de rectas.
