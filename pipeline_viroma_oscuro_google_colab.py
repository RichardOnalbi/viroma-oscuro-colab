"""
Pipeline para Google Colab
Articulo: viroma oscuro de Culicidae neotropicales

Uso recomendado:
1. Abrir un notebook nuevo en Google Colab.
2. Copiar cada bloque marcado con "# %% [markdown]" o "# %%" en celdas separadas.
3. Ejecutar primero el flujo ligero: dataset curado, tablas y figuras.
4. Ejecutar el modulo SRA/DIAMOND solo si se requiere reprocesar lecturas publicas.

Regla editorial:
Este pipeline no fabrica cifras. Distingue datos exactos publicados, evidencia
cualitativa y analisis prospectivos. Las figuras y tablas deben conservar esa
distincion en el manuscrito.
"""

# %% [markdown]
# # Viroma oscuro de Culicidae neotropicales
#
# Pipeline Colab para obtener y organizar datos segun el objetivo, alcance y
# tematica del articulo.
#
# **Objetivo operativo**
# - Consolidar un corpus de estudios neotropicales sobre viromica/metagenomica
#   de mosquitos Culicidae.
# - Separar evidencia cuantitativa exacta de evidencia cualitativa.
# - Generar insumos para el manuscrito: dataset curado, tablas, figuras y
#   resumen tecnico para escritura posterior.
# - Dejar preparado un modulo opcional para reprocesar secuencias publicas
#   desde SRA con Fastp, MEGAHIT, Prodigal y DIAMOND.
#
# **Alcance**
# - Colombia y Brasil como foco neotropical inicial.
# - Estudios 2021-2026 incluidos en el documento de arquitectura.
# - Enfoque en fraccion no clasificada, virus divergentes, ISVs y limite del
#   paradigma por homologia.
# - LucaProt se trata como perspectiva metodologica, no como resultado aplicado
#   a Culicidae neotropicales salvo que se ejecute un analisis nuevo.

# %%
# @title 1. Configuracion de Colab y dependencias
RUN_IN_COLAB = False
try:
    import google.colab  # type: ignore
    RUN_IN_COLAB = True
except Exception:
    RUN_IN_COLAB = False

if RUN_IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

# En Colab, ejecuta esta celda si faltan paquetes:
# !pip -q install pandas numpy matplotlib seaborn openpyxl biopython
# Opcional para mapas mas elaborados:
# !pip -q install geopandas shapely pyogrio

from pathlib import Path
import json
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

BASE_DIR = Path("/content/viroma_oscuro_colab") if RUN_IN_COLAB else Path("viroma_oscuro_colab")
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figuras"
TAB_DIR = OUT_DIR / "tablas"
DATA_DIR = BASE_DIR / "data"

for directory in [BASE_DIR, OUT_DIR, FIG_DIR, TAB_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Directorio base: {BASE_DIR.resolve()}")

# %%
# @title 2. Parametros editoriales del articulo
ARTICLE_SCOPE = {
    "tipo": "articulo de reflexion",
    "revista": "Acta Biologica Colombiana",
    "tema": "viroma oscuro de Culicidae neotropicales",
    "argumento_central": (
        "La fraccion no clasificada no es solo ruido tecnico ni falta de "
        "profundidad de secuenciacion; refleja limites del analisis por "
        "homologia frente a virus altamente divergentes."
    ),
    "foco_geografico": ["Colombia", "Brasil"],
    "comparacion_metodologica": [
        "DIAMOND/BLAST + MEGAN/MEGANizer",
        "modelos de lenguaje proteico / LucaProt como perspectiva",
    ],
    "productos": [
        "dataset bibliografico curado",
        "Tabla 1: herramientas y limites metodologicos",
        "Tabla 2: estudios/ecosistemas y brechas de deteccion",
        "Figura 1: mapa + barras de evidencia neotropical",
        "Figura 2: pipeline convencional vs aumentado con IA",
        "Figura 3: comparativo del viroma oscuro",
        "resumen tecnico para Claude",
    ],
}

with open(OUT_DIR / "scope_articulo.json", "w", encoding="utf-8") as fh:
    json.dump(ARTICLE_SCOPE, fh, indent=2, ensure_ascii=False)

ARTICLE_SCOPE

# %%
# @title 3. Paleta editorial y utilidades
PALETTE = {
    "paper": "#f5f2eb",
    "ink": "#0f1117",
    "green": "#1a4a2e",
    "amber": "#b85c2a",
    "blue": "#2c5f8a",
    "mid": "#6b6560",
    "light": "#e0dbd0",
    "light_g": "#d4e8d8",
    "light_b": "#d0dff0",
    "red_box": "#e8c4b8",
    "colombia": "#1a4a2e",
    "brasil": "#b85c2a",
    "other": "#6b6560",
}

CMAP_DARK = LinearSegmentedColormap.from_list(
    "dark_virome", ["#4a9e6e", "#e8b84b", "#c0392b"]
)

plt.rcParams.update({
    "figure.facecolor": PALETTE["paper"],
    "axes.facecolor": PALETTE["paper"],
    "savefig.facecolor": PALETTE["paper"],
    "font.size": 9,
    "axes.edgecolor": PALETTE["light"],
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["mid"],
    "ytick.color": PALETTE["mid"],
})


def save_figure(fig, stem: str):
    """Exporta cada figura como PNG y PDF."""
    outputs = []
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=PALETTE["paper"])
        outputs.append(out)
    print("Exportado:", " | ".join(str(x) for x in outputs))
    return outputs


def wrap_label(value, width=32):
    return "\n".join(textwrap.wrap(str(value), width=width))

# %%
# @title 4. Dataset curado de estudios neotropicales
def build_neotropical_dataset() -> pd.DataFrame:
    """Dataset semilla basado en la arquitectura del articulo.

    Campos clave:
    - data_type: exact, partial, qualitative.
    - dark_score: valor graficable. Para datos cualitativos es un indice visual,
      no un porcentaje citable.
    - citable_dark_pct: porcentaje solo cuando esta publicado o directamente
      calculado desde una cifra publicada.
    """
    records = [
        {
            "ref_short": "Hoyos-Lopez 2025",
            "authors_year": "Hoyos-Lopez et al., 2025",
            "country": "Colombia",
            "region": "Caribe colombiano: Cordoba, Sucre, Bolivar, Magdalena",
            "lat": 8.75,
            "lon": -75.88,
            "journal": "Memorias do Instituto Oswaldo Cruz",
            "doi": "10.1590/0074-02760250131",
            "sra": "PRJNA1251058",
            "ecosystem": "humedales, manglares y zonas agropecuarias",
            "genera": "Mansonia, Coquillettidia, Anopheles, Culex",
            "n_mosquitoes": 4074,
            "n_pools": 33,
            "pipeline": "DIAMOND-MEGANizer vs NCBI-nr",
            "viral_families": 22,
            "dark_virome_n": 24,
            "dark_richness_pct": 44.7,
            "citable_dark_pct": 44.7,
            "data_type": "exact",
            "dark_score": 44.7,
            "key_message": "24 virus ARN sin clasificacion ICTV; Ma. titillans 17/38 = 44.7%.",
            "use_in_article": "evidencia primaria colombiana y distincion entre reads no clasificados y virus sin hogar taxonomico",
        },
        {
            "ref_short": "Hernandez-Valencia 2025",
            "authors_year": "Hernandez-Valencia et al., 2025",
            "country": "Colombia",
            "region": "tres ecorregiones colombianas",
            "lat": 5.50,
            "lon": -73.50,
            "journal": "PLOS ONE",
            "doi": "10.1371/journal.pone.0320593",
            "sra": "",
            "ecosystem": "ecorregiones contrastantes con Anopheles darlingi",
            "genera": "Anopheles",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "metagenomica viral",
            "viral_families": np.nan,
            "dark_virome_n": np.nan,
            "dark_richness_pct": np.nan,
            "citable_dark_pct": 55.8,
            "data_type": "exact",
            "dark_score": 55.8,
            "key_message": "48.9% Ortervirales no clasificados + 6.9% virus no clasificados.",
            "use_in_article": "comparativo colombiano con cifra exacta de fraccion no clasificada",
        },
        {
            "ref_short": "Gomez-Palacio 2025",
            "authors_year": "Gomez-Palacio et al., 2025",
            "country": "Colombia",
            "region": "valles interandinos del norte",
            "lat": 6.45,
            "lon": -75.60,
            "journal": "PLOS ONE",
            "doi": "10.1371/journal.pone.0331552",
            "sra": "PRJNA1199888",
            "ecosystem": "ambientes no urbanos",
            "genera": "Culicinae",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "metagenomica + clasificacion viral",
            "viral_families": np.nan,
            "dark_virome_n": np.nan,
            "dark_richness_pct": np.nan,
            "citable_dark_pct": np.nan,
            "data_type": "qualitative",
            "dark_score": 65.0,
            "key_message": "La mayoria de contigs virales de alta confianza permanecio sin clasificar.",
            "use_in_article": "evidencia cualitativa de persistencia del viroma oscuro en Colombia",
        },
        {
            "ref_short": "Gomez 2023",
            "authors_year": "Gomez et al., 2023",
            "country": "Colombia",
            "region": "Orinoquia colombiana",
            "lat": 4.90,
            "lon": -71.60,
            "journal": "Scientific Reports",
            "doi": "10.1038/s41598-023-49232-9",
            "sra": "",
            "ecosystem": "Orinoco",
            "genera": "Culicidae",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "metagenomica viral",
            "viral_families": np.nan,
            "dark_virome_n": np.nan,
            "dark_richness_pct": np.nan,
            "citable_dark_pct": np.nan,
            "data_type": "partial",
            "dark_score": 35.0,
            "key_message": "ISVs dominantes; fraccion no clasificada no reportada explicitamente.",
            "use_in_article": "contexto regional colombiano y co-circulacion de ISVs",
        },
        {
            "ref_short": "Aragao 2023",
            "authors_year": "Aragao et al., 2023",
            "country": "Brasil",
            "region": "transicion Amazonia-Cerrado-Caatinga",
            "lat": -5.10,
            "lon": -45.00,
            "journal": "Genes",
            "doi": "10.3390/genes14071443",
            "sra": "",
            "ecosystem": "ecotono norte-nordeste",
            "genera": "Culicidae",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "shotgun metagenomics",
            "viral_families": np.nan,
            "dark_virome_n": 2,
            "dark_richness_pct": np.nan,
            "citable_dark_pct": np.nan,
            "data_type": "qualitative",
            "dark_score": 70.0,
            "key_message": "15 genomas virus-like, todos divergentes de referencias; 2 contigs Riboviria sp.",
            "use_in_article": "ejemplo brasileno de divergencia generalizada frente a referencias",
        },
        {
            "ref_short": "Da Silva 2024",
            "authors_year": "Da Silva et al., 2024",
            "country": "Brasil",
            "region": "nordeste de Brasil",
            "lat": -8.20,
            "lon": -36.50,
            "journal": "Journal of Virology",
            "doi": "10.1128/jvi.00083-24",
            "sra": "",
            "ecosystem": "mosquitos silvestres",
            "genera": "Culicidae",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "DIAMOND + NeoRdRp + PalmDB + RVMT + RdRp-Scan",
            "viral_families": 16,
            "dark_virome_n": np.nan,
            "dark_richness_pct": np.nan,
            "citable_dark_pct": np.nan,
            "data_type": "qualitative",
            "dark_score": 62.0,
            "key_message": "Alta diversidad; mayoria de secuencias corresponde a nuevas especies virales.",
            "use_in_article": "evidencia brasilena de diversidad viral altamente divergente",
        },
        {
            "ref_short": "Maia 2024",
            "authors_year": "Maia et al., 2024",
            "country": "Brasil",
            "region": "Cerrado, Minas Gerais",
            "lat": -18.50,
            "lon": -44.00,
            "journal": "Viruses",
            "doi": "10.3390/v16081276",
            "sra": "",
            "ecosystem": "Cerrado",
            "genera": "Culicidae",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "metagenomica / metatranscriptomica viral",
            "viral_families": 7,
            "dark_virome_n": 7,
            "dark_richness_pct": 63.6,
            "citable_dark_pct": 63.6,
            "data_type": "exact",
            "dark_score": 63.6,
            "key_message": "7 de 11 virus casi completos fueron nuevos.",
            "use_in_article": "comparativo brasileno cuantificable de especies nuevas",
        },
        {
            "ref_short": "Da Silva 2021",
            "authors_year": "Da Silva et al., 2021",
            "country": "Brasil",
            "region": "neotropico brasileno",
            "lat": -15.00,
            "lon": -47.00,
            "journal": "Virus Research",
            "doi": "10.1016/j.virusres.2021.198455",
            "sra": "",
            "ecosystem": "Mansoniini",
            "genera": "Mansonia, Coquillettidia",
            "n_mosquitoes": np.nan,
            "n_pools": np.nan,
            "pipeline": "metatranscriptomics + DIAMOND",
            "viral_families": np.nan,
            "dark_virome_n": np.nan,
            "dark_richness_pct": np.nan,
            "citable_dark_pct": np.nan,
            "data_type": "partial",
            "dark_score": 40.0,
            "key_message": "Secuencias viral-like divergentes en Mansoniini.",
            "use_in_article": "antecedente para generos focales del Caribe colombiano",
        },
    ]

    df = pd.DataFrame(records)
    df["is_exact"] = df["data_type"].eq("exact")
    df["has_sra"] = df["sra"].fillna("").ne("")
    df["country_order"] = df["country"].map({"Colombia": 0, "Brasil": 1}).fillna(9)
    return df


df = build_neotropical_dataset()
df

# %%
# @title 5. Exportar dataset y tablas base
dataset_out = DATA_DIR / "neotropical_virome_dataset_curated.csv"
df.to_csv(dataset_out, index=False, encoding="utf-8")

table2_cols = [
    "ref_short", "country", "region", "ecosystem", "genera", "viral_families",
    "dark_virome_n", "citable_dark_pct", "data_type", "key_message", "doi", "sra",
]
table2 = df[table2_cols].copy()
table2.columns = [
    "Referencia", "Pais", "Region", "Ecosistema", "Generos", "Familias virales",
    "Virus/contigs sin clasificacion", "Porcentaje citable", "Tipo de dato",
    "Mensaje clave", "DOI", "SRA",
]
table2_out = TAB_DIR / "Tabla_2_estudios_neotropicales.csv"
table2.to_csv(table2_out, index=False, encoding="utf-8")

table1 = pd.DataFrame([
    {
        "Herramienta": "BLAST / DIAMOND",
        "Principio": "homologia de secuencia",
        "Fortaleza": "rapido, reproducible y preciso cuando hay referencias cercanas",
        "Limite": "pierde sensibilidad frente a virus altamente divergentes o no representados",
        "Uso en articulo": "paradigma convencional aplicado en estudios neotropicales",
    },
    {
        "Herramienta": "MEGAN / MEGANizer",
        "Principio": "asignacion taxonomica por LCA",
        "Fortaleza": "resume lecturas/contigs en categorias taxonomicas interpretables",
        "Limite": "depende de hits previos y de taxonomia disponible",
        "Uso en articulo": "explica la distincion entre read clasificado y virus sin clasificacion ICTV",
    },
    {
        "Herramienta": "HMMscan / perfiles RdRP",
        "Principio": "motivos conservados y perfiles de familias proteicas",
        "Fortaleza": "mejor sensibilidad que alineamiento simple para dominios conservados",
        "Limite": "requiere perfiles entrenados y puede fallar ante supergrupos nuevos",
        "Uso en articulo": "puente entre homologia tradicional y modelos semanticos",
    },
    {
        "Herramienta": "LucaProt",
        "Principio": "modelo de lenguaje proteico sobre RdRP",
        "Fortaleza": "detecta senales semanticas/funcionales no capturadas por homologia directa",
        "Limite": "demanda computacional y falta validacion especifica en Culicidae neotropicales",
        "Uso en articulo": "perspectiva metodologica, no sustituto automatico de validacion experimental",
    },
])

table1_out = TAB_DIR / "Tabla_1_herramientas_bioinformaticas.csv"
table1.to_csv(table1_out, index=False, encoding="utf-8")

print(f"Dataset: {dataset_out}")
print(f"Tabla 1: {table1_out}")
print(f"Tabla 2: {table2_out}")

# %%
# @title 6. Figura 1: mapa bibliometrico simple + barras
def plot_figura1(df: pd.DataFrame):
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_map.set_title("A. Estudios neotropicales incluidos", color=PALETTE["green"], fontsize=11)
    ax_map.set_xlim(-82, -34)
    ax_map.set_ylim(-25, 14)
    ax_map.set_xlabel("Longitud")
    ax_map.set_ylabel("Latitud")
    ax_map.grid(color=PALETTE["light"], linestyle=":", linewidth=0.8)

    for _, row in df.iterrows():
        color = PALETTE["colombia"] if row["country"] == "Colombia" else PALETTE["brasil"]
        size = 80 + 18 * (0 if pd.isna(row["viral_families"]) else row["viral_families"])
        marker = "o" if row["data_type"] == "exact" else "s"
        ax_map.scatter(row["lon"], row["lat"], s=size, c=color, marker=marker,
                       alpha=0.82, edgecolor="white", linewidth=0.8)
        ax_map.text(row["lon"] + 0.8, row["lat"] + 0.4, row["ref_short"],
                    fontsize=7, color=PALETTE["ink"])

    legend_handles = [
        mpatches.Patch(color=PALETTE["colombia"], label="Colombia"),
        mpatches.Patch(color=PALETTE["brasil"], label="Brasil"),
    ]
    ax_map.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=8)

    df_plot = df.sort_values(["country_order", "dark_score"], ascending=[True, False])
    colors = [PALETTE["colombia"] if x == "Colombia" else PALETTE["brasil"] for x in df_plot["country"]]
    hatches = ["" if x == "exact" else "///" for x in df_plot["data_type"]]
    alphas = [0.9 if x == "exact" else 0.42 for x in df_plot["data_type"]]

    y = np.arange(len(df_plot))
    for i, (_, row) in enumerate(df_plot.iterrows()):
        bar = ax_bar.barh(i, row["dark_score"], color=colors[i], alpha=alphas[i],
                          edgecolor=PALETTE["ink"], linewidth=0.3)
        bar[0].set_hatch(hatches[i])
        label = f"{row['citable_dark_pct']:.1f}%" if pd.notna(row["citable_dark_pct"]) else "cualitativo"
        ax_bar.text(row["dark_score"] + 1, i, label, va="center", fontsize=7, color=PALETTE["mid"])

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(df_plot["ref_short"], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 82)
    ax_bar.set_xlabel("Fraccion no clasificada / indice visual (%)")
    ax_bar.set_title("B. Evidencia de viroma oscuro", color=PALETTE["green"], fontsize=11)
    ax_bar.grid(axis="x", color=PALETTE["light"], linestyle=":", linewidth=0.8)

    exact_patch = mpatches.Patch(facecolor=PALETTE["mid"], alpha=0.9, label="dato exacto")
    qual_patch = mpatches.Patch(facecolor=PALETTE["mid"], alpha=0.42, hatch="///",
                                label="evidencia cualitativa")
    ax_bar.legend(handles=[exact_patch, qual_patch], frameon=False, loc="lower right", fontsize=8)

    fig.suptitle(
        "Figura 1. Diversidad viral y fraccion no clasificada en Culicidae neotropicales",
        fontsize=13, color=PALETTE["green"], y=0.98
    )
    return fig


fig = plot_figura1(df)
save_figure(fig, "Figura1_mapa_bibliometrico_neotropical")
plt.show()

# %%
# @title 7. Figura 2: pipeline convencional vs pipeline aumentado con IA
def draw_box(ax, x, y, w, h, text, bg, fc=None, fs=7.5):
    fc = fc or PALETTE["ink"]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=0.8, edgecolor=PALETTE["mid"], facecolor=bg
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=fc)


def draw_arrow(ax, x, y1, y2, color=None):
    arrow = FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.1, color=color or PALETTE["mid"]
    )
    ax.add_patch(arrow)


def plot_figura2():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(2.5, 9.55, "Pipeline convencional\n(aplicado)", ha="center",
            fontsize=11, color=PALETTE["blue"], weight="bold")
    ax.text(7.5, 9.55, "Pipeline aumentado con IA\n(prospectivo)", ha="center",
            fontsize=11, color=PALETTE["green"], weight="bold")

    left_steps = [
        "Captura de mosquitos\nCulicidae",
        "Extraccion de RNA/DNA\n+ secuenciacion NGS",
        "Control de calidad\nFastp / Trimmomatic",
        "Ensamblaje de novo\nMEGAHIT / Trinity",
        "ORFs / proteinas\nProdigal / ORFfinder",
        "Homologia\nDIAMOND/BLAST vs nr",
        "Asignacion taxonomica\nMEGAN / MEGANizer",
        "Fraccion no clasificada\nvirus divergentes",
    ]
    right_steps = [
        "Mismos datos iniciales\nSRA / datos propios",
        "Control de calidad\n+ ensamblaje",
        "ORFs / proteinas\nfasta aminoacidos",
        "Deteccion RdRP\nperfiles + filtros",
        "Embeddings proteicos\nLucaProt / transformer",
        "Similitud semantica\n+ supergrupos virales",
        "Priorizacion de candidatos\npara validacion",
        "Reduccion interpretativa\ndel viroma oscuro",
    ]

    y_positions = [8.45, 7.45, 6.45, 5.45, 4.45, 3.45, 2.45, 1.45]
    for i, (step, y) in enumerate(zip(left_steps, y_positions)):
        bg = PALETTE["red_box"] if i == len(left_steps) - 1 else PALETTE["light_b"]
        draw_box(ax, 0.65, y, 3.65, 0.55, step, bg, fs=7.2)
        if i < len(left_steps) - 1:
            draw_arrow(ax, 2.48, y, y - 0.42, PALETTE["blue"])

    for i, (step, y) in enumerate(zip(right_steps, y_positions)):
        bg = PALETTE["light_g"] if i < len(right_steps) - 1 else "#cce5cc"
        draw_box(ax, 5.70, y, 3.65, 0.55, step, bg, fs=7.2)
        if i < len(right_steps) - 1:
            draw_arrow(ax, 7.52, y, y - 0.42, PALETTE["green"])

    ax.text(
        5.0, 0.42,
        "Nota: la columna izquierda resume experiencia documentada; la derecha es una ruta prospectiva que requiere validacion.",
        ha="center", fontsize=7.5, color=PALETTE["mid"], style="italic"
    )
    ax.set_title(
        "Figura 2. Del analisis por homologia al paradigma semantico en metagenomica viral",
        fontsize=12, color=PALETTE["green"], pad=8
    )
    return fig


fig = plot_figura2()
save_figure(fig, "Figura2_pipeline_convencional_vs_IA")
plt.show()

# %%
# @title 8. Figura 3: comparativo del viroma oscuro
def plot_figura3(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.16)
    df_plot = df.sort_values("dark_score", ascending=True)

    y = np.arange(len(df_plot))
    colors = [PALETTE["colombia"] if c == "Colombia" else PALETTE["brasil"] for c in df_plot["country"]]

    for i, (_, row) in enumerate(df_plot.iterrows()):
        hatch = "" if row["data_type"] == "exact" else "///"
        alpha = 0.88 if row["data_type"] == "exact" else 0.42
        bar = ax.barh(i, row["dark_score"], color=colors[i], alpha=alpha,
                      edgecolor=PALETTE["ink"], linewidth=0.35)
        bar[0].set_hatch(hatch)
        label = "exacto" if row["data_type"] == "exact" else row["data_type"]
        ax.text(row["dark_score"] + 1, i, label, va="center", fontsize=7, color=PALETTE["mid"])

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["ref_short"], fontsize=8)
    ax.set_xlim(0, 82)
    ax.set_xlabel("Porcentaje citable o indice visual de evidencia")
    ax.set_title(
        "Figura 3. Patron regional del viroma oscuro en estudios Colombia-Brasil",
        color=PALETTE["green"], fontsize=12
    )
    ax.grid(axis="x", color=PALETTE["light"], linestyle=":", linewidth=0.8)

    note = (
        "Barras solidas: porcentaje publicado o calculado desde cifras publicadas. "
        "Barras rayadas: evidencia cualitativa; no comparar como porcentaje real."
    )
    fig.text(0.13, 0.045, note, fontsize=7.5, color=PALETTE["mid"], style="italic")

    handles = [
        mpatches.Patch(color=PALETTE["colombia"], label="Colombia"),
        mpatches.Patch(color=PALETTE["brasil"], label="Brasil"),
        mpatches.Patch(facecolor=PALETTE["mid"], alpha=0.4, hatch="///", label="cualitativo"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    return fig


fig = plot_figura3(df)
save_figure(fig, "Figura3_viroma_oscuro_comparativo")
plt.show()

# %%
# @title 9. Analisis de salida DIAMOND/MEGANizer si ya existe diamond_hits.tsv
def analyze_diamond_output(diamond_tsv: str, total_orfs: int, sample_name: str):
    """Cuantifica ORFs con hits virales conocidos, virales oscuros y sin hit.

    Entrada esperada de DIAMOND:
    qseqid sseqid pident length evalue bitscore staxids sscinames
    """
    diamond_path = Path(diamond_tsv)
    if not diamond_path.exists():
        print(f"[opcional] No se encontro {diamond_tsv}.")
        print("Este modulo solo se ejecuta cuando ya tienes una salida DIAMOND.")
        return None, None

    cols = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "staxids", "sscinames"]
    hits = pd.read_csv(diamond_path, sep="\t", names=cols, comment="#")
    hits["category"] = "non_viral"

    names = hits["sscinames"].fillna("").str.lower()
    viral_mask = names.str.contains("virus|viridae|viricetes|phage|viricota|riboviria", regex=True)
    unclass_mask = names.str.contains("unclassified|uncharacterized|hypothetical|unnamed|unknown", regex=True)

    hits.loc[viral_mask & ~unclass_mask, "category"] = "viral_known"
    hits.loc[viral_mask & unclass_mask, "category"] = "dark_viral"
    hits.loc[~viral_mask & unclass_mask, "category"] = "hypothetical"

    n_with_hit = hits["qseqid"].nunique()
    n_no_hit = max(0, int(total_orfs) - int(n_with_hit))
    counts = hits["category"].value_counts().to_dict()
    counts["no_hit"] = n_no_hit

    dark_n = counts.get("dark_viral", 0) + counts.get("no_hit", 0)
    dark_pct = round(100 * dark_n / total_orfs, 2) if total_orfs else np.nan

    result = {
        "sample": sample_name,
        "total_orfs": total_orfs,
        "orfs_with_hit": n_with_hit,
        "viral_known": counts.get("viral_known", 0),
        "dark_viral": counts.get("dark_viral", 0),
        "hypothetical": counts.get("hypothetical", 0),
        "non_viral": counts.get("non_viral", 0),
        "no_hit": n_no_hit,
        "dark_matter_n": dark_n,
        "dark_matter_pct": dark_pct,
    }
    return result, hits


# Ejemplo:
# diamond_file = DATA_DIR / "diamond_hits.tsv"
# result, hits = analyze_diamond_output(diamond_file, total_orfs=15000, sample_name="PRJNA1251058_pool_X")
# if result is not None:
#     pd.DataFrame([result]).to_csv(TAB_DIR / "resultado_diamond_pool_X.csv", index=False)
#     result

# %% [markdown]
# ## 10. Modulo opcional pesado: SRA -> FASTQ -> ensamblaje -> DIAMOND
#
# Esta parte no es necesaria para las figuras y tablas bibliograficas que ya
# generaste. Solo se usa si quieres reprocesar secuencias publicas o propias.
#
# En Colab gratuito puede tardar horas y consumir mucho disco. Por eso queda
# apagada por defecto. Para ejecutarla, cambia `RUN_SRA_PIPELINE = False` a
# `RUN_SRA_PIPELINE = True` y reemplaza `SRRXXXXXXX` por un acceso real.
#
# Nota importante: DIAMOND requiere una base `.dmnd`. La base `nr` completa no
# es realista en Colab gratuito por tamano; conviene usar una base viral reducida
# preparada previamente, o ejecutar esa parte en HPC/servidor.

# %%
# @title 10A. Instalar herramientas del modulo SRA/DIAMOND (opcional)
RUN_SRA_PIPELINE = False  # Cambiar a True solo si vas a reprocesar lecturas

if RUN_SRA_PIPELINE:
    if not RUN_IN_COLAB:
        print("Este bloque esta pensado para Google Colab/Linux.")
    else:
        get_ipython().system("apt-get update -qq")
        get_ipython().system("apt-get install -y -qq sra-toolkit fastp megahit prodigal diamond-aligner")
        print("Herramientas instaladas.")
else:
    print("Modulo SRA/DIAMOND omitido. Cambia RUN_SRA_PIPELINE a True para activarlo.")

# %%
# @title 10B. Descargar FASTQ desde SRA (opcional)
SRR_ACCESSION = "SRRXXXXXXX"  # Reemplazar por un SRR real
SRA_DIR = DATA_DIR / "sra"
SRA_DIR.mkdir(parents=True, exist_ok=True)

if RUN_SRA_PIPELINE:
    get_ipython().system(f"fasterq-dump {SRR_ACCESSION} -O {SRA_DIR} --split-files --threads 4")
    get_ipython().system(f"gzip -f {SRA_DIR}/{SRR_ACCESSION}_*.fastq")
    print(f"FASTQ descargados en: {SRA_DIR}")
else:
    print("Descarga SRA omitida.")

# %%
# @title 10C. Control de calidad con fastp (opcional)
if RUN_SRA_PIPELINE:
    read1 = SRA_DIR / f"{SRR_ACCESSION}_1.fastq.gz"
    read2 = SRA_DIR / f"{SRR_ACCESSION}_2.fastq.gz"
    clean1 = DATA_DIR / f"{SRR_ACCESSION}.clean_1.fastq.gz"
    clean2 = DATA_DIR / f"{SRR_ACCESSION}.clean_2.fastq.gz"
    html = OUT_DIR / f"{SRR_ACCESSION}_fastp.html"

    get_ipython().system(
        f"fastp -i {read1} -I {read2} -o {clean1} -O {clean2} "
        f"--qualified_quality_phred 20 --length_required 50 --thread 4 --html {html}"
    )
    print(f"Lecturas limpias: {clean1}, {clean2}")
else:
    print("fastp omitido.")

# %%
# @title 10D. Ensamblaje con MEGAHIT y prediccion de ORFs (opcional)
if RUN_SRA_PIPELINE:
    clean1 = DATA_DIR / f"{SRR_ACCESSION}.clean_1.fastq.gz"
    clean2 = DATA_DIR / f"{SRR_ACCESSION}.clean_2.fastq.gz"
    assembly_dir = DATA_DIR / f"{SRR_ACCESSION}_megahit"
    proteins_faa = DATA_DIR / f"{SRR_ACCESSION}.proteins.faa"

    get_ipython().system(
        f"megahit -1 {clean1} -2 {clean2} -o {assembly_dir} "
        "--min-contig-len 300 --num-cpu-threads 4"
    )

    get_ipython().system(
        f"prodigal -i {assembly_dir}/final.contigs.fa -a {proteins_faa} -p meta"
    )

    total_orfs_result = get_ipython().getoutput(f'grep -c ">" {proteins_faa}')
    total_orfs = int(total_orfs_result[0]) if total_orfs_result else 0
    print(f"Proteinas predichas: {proteins_faa}")
    print(f"ORFs totales: {total_orfs}")
else:
    print("MEGAHIT/Prodigal omitidos.")

# %%
# @title 10E. DIAMOND contra base viral o nr reducida (opcional)
DIAMOND_DB = "/content/path/to/viral_or_nr.dmnd"  # Reemplazar por tu base .dmnd

if RUN_SRA_PIPELINE:
    proteins_faa = DATA_DIR / f"{SRR_ACCESSION}.proteins.faa"
    diamond_out = DATA_DIR / f"{SRR_ACCESSION}.diamond_hits.tsv"

    if not Path(DIAMOND_DB).exists():
        print(f"No existe la base DIAMOND: {DIAMOND_DB}")
        print("Sube o prepara una base .dmnd antes de ejecutar esta celda.")
    else:
        get_ipython().system(
            f"diamond blastp -q {proteins_faa} -d {DIAMOND_DB} -o {diamond_out} "
            "--outfmt 6 qseqid sseqid pident length evalue bitscore staxids sscinames "
            "--evalue 1e-5 --max-target-seqs 1 --sensitive --threads 4"
        )

        total_orfs_result = get_ipython().getoutput(f'grep -c ">" {proteins_faa}')
        total_orfs = int(total_orfs_result[0]) if total_orfs_result else 0
        result, hits = analyze_diamond_output(diamond_out, total_orfs, SRR_ACCESSION)
        if result is not None:
            pd.DataFrame([result]).to_csv(TAB_DIR / f"{SRR_ACCESSION}_diamond_summary.csv", index=False)
            print(result)
else:
    print("DIAMOND omitido.")

# %%
# @title 11. Resumen tecnico para Claude / manuscrito
def build_manuscript_brief(df: pd.DataFrame) -> str:
    exact = df[df["data_type"].eq("exact")]
    qualitative = df[~df["data_type"].eq("exact")]

    lines = [
        "# Brief tecnico para manuscrito",
        "",
        "## Tesis",
        ARTICLE_SCOPE["argumento_central"],
        "",
        "## Evidencia cuantitativa citable",
    ]
    for _, row in exact.iterrows():
        lines.append(
            f"- {row['ref_short']} ({row['country']}): {row['key_message']} DOI: {row['doi']}"
        )

    lines.extend([
        "",
        "## Evidencia cualitativa o parcial",
    ])
    for _, row in qualitative.iterrows():
        lines.append(
            f"- {row['ref_short']} ({row['country']}): {row['key_message']} DOI: {row['doi']}"
        )

    lines.extend([
        "",
        "## Precaucion metodologica",
        (
            "No equiparar reads sin clasificar con viroma oscuro. En el argumento "
            "del articulo, el punto central es la existencia de virus/contigs con "
            "senal viral pero sin clasificacion taxonomica formal o sin referencia "
            "cercana suficiente."
        ),
        "",
        "## LucaProt",
        (
            "Presentarlo como perspectiva metodologica: un cambio desde homologia "
            "lineal hacia representaciones semanticas de proteinas virales. Aclarar "
            "que su aplicacion a Culicidae neotropicales debe validarse con datos "
            "propios o publicos y no se debe reportar como resultado si no se ejecuta."
        ),
        "",
        "## Archivos generados",
        f"- Dataset curado: {dataset_out}",
        f"- Tabla 1: {table1_out}",
        f"- Tabla 2: {table2_out}",
        f"- Figuras: {FIG_DIR}",
    ])
    return "\n".join(lines)


brief = build_manuscript_brief(df)
brief_out = OUT_DIR / "brief_tecnico_para_claude.md"
brief_out.write_text(brief, encoding="utf-8")
print(f"Brief exportado: {brief_out}")
try:
    from IPython.display import Markdown, display
    display(Markdown(brief))
except Exception:
    print(brief[:1200])

# %%
# @title 12. Empaquetar salidas para descarga desde Colab
if RUN_IN_COLAB:
    import shutil
    zip_base = shutil.make_archive(str(BASE_DIR / "viroma_oscuro_outputs"), "zip", root_dir=OUT_DIR)
    print(f"ZIP listo: {zip_base}")
    # from google.colab import files
    # files.download(zip_base)
else:
    print(f"Salidas locales en: {OUT_DIR.resolve()}")
