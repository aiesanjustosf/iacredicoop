# ia_resumen_credicoop.py
# IA Resumen Credicoop — AIE San Justo
# Parser robusto (CHARS + OCR fallback) para Resumen de Cuenta Corriente Comercial (Banco Credicoop).
#
# Reglas clave:
# - Conciliación estricta: Saldo anterior + Créditos − Débitos = Saldo al <fecha>
# - El saldo que aparece en la columna SALDO a mitad del listado NO se usa (es saldo diario).
#   Se usa exclusivamente:
#     * "SALDO ANTERIOR"  -> saldo inicial
#     * "SALDO AL <dd/mm/aa|aaaa>" -> saldo final
# - Débito y Crédito se determinan por columna (nunca se infiere por texto).

from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# deps
try:
    import pdfplumber
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"No se pudo importar pdfplumber: {e}") from e

try:
    import pytesseract
except Exception as e:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"No se pudo importar Pillow: {e}") from e

try:
    import pypdfium2 as pdfium
except Exception as e:  # pragma: no cover
    pdfium = None

# PDF (Resumen Operativo)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ------------------------------
# Config / UI assets
# ------------------------------
HERE = Path(__file__).parent
LOGO = (HERE / "assets" / "logo_aie.png") if (HERE / "assets" / "logo_aie.png").exists() else (HERE / "logo_aie.png")
FAVICON = (HERE / "assets" / "favicon-aie.ico") if (HERE / "assets" / "favicon-aie.ico").exists() else (HERE / "favicon-aie.ico")


# ------------------------------
# Utilidades
# ------------------------------
DATE_ANY_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
DATE_AT_START_RE = re.compile(r"^\s*(\d{2}/\d{2}/\d{2,4})\b")

# pesos argentinos típicos: 1.234,56 / 1234,56 y con signo -
MONEY_RE = re.compile(r"(?P<sign>[-−])?\s*(?P<num>(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2})(?P<trailing>-)?")

def _strip_accents(s: str) -> str:
    # sin dependencia extra
    repl = str.maketrans(
        "ÁÉÍÓÚÜÑáéíóúüñ",
        "AEIOUUNaeiouun"
    )
    return s.translate(repl)

def norm_text(s: str) -> str:
    s = _strip_accents(s or "")
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_money(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace(" ", "")
    m = MONEY_RE.search(s)
    if not m:
        return None
    tok = m.group("num")
    sign = m.group("sign") or ""
    trailing = m.group("trailing") or ""
    neg = (sign in ("-", "−")) or (trailing == "-")
    main, frac = tok.split(",")
    main = main.replace(".", "")
    try:
        val = float(f"{main}.{frac}")
    except Exception:
        return None
    return -val if neg else val

def fmt_ar(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return ""
    s = f"{x:,.2f}"
    # 1,234,567.89 -> 1.234.567,89
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


# ------------------------------
# CHARS: reconstrucción de líneas
# ------------------------------
def _group_chars_to_lines(chars: List[dict], y_tol: float = 2.0) -> List[List[dict]]:
    if not chars:
        return []
    # usar "top" (pdfplumber) para agrupar por banda
    chars_sorted = sorted(chars, key=lambda c: (round(c["top"] / y_tol), c["x0"]))
    lines: List[List[dict]] = []
    cur: List[dict] = []
    cur_band = None
    for c in chars_sorted:
        b = round(c["top"] / y_tol)
        if cur_band is None or b == cur_band:
            cur.append(c)
            cur_band = b
        else:
            lines.append(sorted(cur, key=lambda x: x["x0"]))
            cur = [c]
            cur_band = b
    if cur:
        lines.append(sorted(cur, key=lambda x: x["x0"]))
    return lines

def _build_text_from_chars(line_chars: List[dict], x0: float, x1: float) -> str:
    # reconstruye texto con espacios por gap
    chars = [c for c in line_chars if c["x0"] >= x0 and c["x1"] <= x1]
    if not chars:
        return ""
    chars = sorted(chars, key=lambda c: c["x0"])
    widths = [c["x1"] - c["x0"] for c in chars if c["text"].strip()]
    med = statistics.median(widths) if widths else 3.0

    out: List[str] = []
    prev_x1 = None
    for c in chars:
        if prev_x1 is not None:
            gap = c["x0"] - prev_x1
            if gap > max(1.2, med * 0.6):
                out.append(" ")
        out.append(c["text"])
        prev_x1 = c["x1"]
    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_word_span(line_chars: List[dict], word: str) -> Optional[Tuple[float, float]]:
    # tolera espaciado entre letras
    texts = [c["text"] for c in line_chars]
    s = "".join(texts)
    pat = r"\s*".join(map(re.escape, list(word)))
    m = re.search(pat, s, flags=re.IGNORECASE)
    if not m:
        return None
    i0, i1 = m.span()
    xs = [(line_chars[i]["x0"], line_chars[i]["x1"]) for i in range(i0, i1) if texts[i].strip()]
    if not xs:
        return None
    return (min(x[0] for x in xs), max(x[1] for x in xs))

@dataclass
class ColLayout:
    b_desc_deb: float
    b_deb_cre: float
    b_cre_sal: float
    ok: bool

def _detect_layout_from_header(lines: List[List[dict]], page_width: float) -> Tuple[Optional[int], Optional[ColLayout]]:
    """
    Busca la línea que contiene el encabezado de tabla y devuelve:
    - índice de línea del header
    - layout con límites (b_desc_deb, b_deb_cre, b_cre_sal)
    """
    header_idx = None
    header_line = None
    for i, line in enumerate(lines):
        s = "".join(c["text"] for c in line).upper()
        if all(k in s for k in ("FECHA", "DEBITO", "CREDITO", "SALDO")):
            header_idx = i
            header_line = line
            break
    if header_line is None:
        return None, None

    spans = {}
    for w in ("DESCRIPCION", "DEBITO", "CREDITO", "SALDO", "COMBTE", "FECHA"):
        spans[w] = _find_word_span(header_line, w)

    # tolerar ausencia de alguno de los de la izquierda, pero exigir los 4 clave
    if any(spans[w] is None for w in ("DESCRIPCION", "DEBITO", "CREDITO", "SALDO")):
        return header_idx, None

    centers = {w: (spans[w][0] + spans[w][1]) / 2 for w in ("DESCRIPCION", "DEBITO", "CREDITO", "SALDO")}
    b_desc_deb = (centers["DESCRIPCION"] + centers["DEBITO"]) / 2
    b_deb_cre = (centers["DEBITO"] + centers["CREDITO"]) / 2
    b_cre_sal = (centers["CREDITO"] + centers["SALDO"]) / 2

    # sanity
    ok = 0 < b_desc_deb < b_deb_cre < b_cre_sal < page_width
    return header_idx, ColLayout(b_desc_deb=b_desc_deb, b_deb_cre=b_deb_cre, b_cre_sal=b_cre_sal, ok=ok)

def parse_pdf_chars(pdf_bytes: bytes) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[str]]:
    """
    Devuelve movimientos + saldo_inicial (SALDO ANTERIOR) + saldo_final (SALDO AL) + fecha_saldo_final.
    """
    rows: List[dict] = []
    saldo_ini = None
    saldo_fin = None
    saldo_fin_fecha = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            if not page.chars:
                continue

            lines = _group_chars_to_lines(page.chars, y_tol=2.0)
            header_idx, layout = _detect_layout_from_header(lines, page.width)
            if header_idx is None or layout is None or not layout.ok:
                continue

            cur = None
            for line in lines[header_idx + 1 :]:
                left = _build_text_from_chars(line, 0, layout.b_desc_deb)
                debs = _build_text_from_chars(line, layout.b_desc_deb, layout.b_deb_cre)
                cres = _build_text_from_chars(line, layout.b_deb_cre, layout.b_cre_sal)
                sals = _build_text_from_chars(line, layout.b_cre_sal, page.width)

                if not (left or debs or cres or sals):
                    continue

                u = norm_text(left)

                # saldos (solo los estructurales)
                if "SALDO ANTERIOR" in u:
                    m = parse_money(sals) or parse_money(cres) or parse_money(debs) or parse_money(left)
                    if m is not None:
                        saldo_ini = m
                    continue

                if "SALDO AL" in u:
                    m = parse_money(sals) or parse_money(cres) or parse_money(debs) or parse_money(left)
                    if m is not None:
                        saldo_fin = m
                    dm = DATE_ANY_RE.search(left)
                    if dm:
                        saldo_fin_fecha = dm.group(1)
                    continue

                dm = DATE_AT_START_RE.search(left)
                if dm:
                    # cierre movimiento anterior
                    if cur:
                        rows.append(cur)

                    fecha_raw = dm.group(1)
                    rest = left[dm.end() :].strip()

                    # comprobante numérico si existe
                    cm = re.match(r"^(\d+)\s+(.*)$", rest)
                    comprobante = None
                    desc = rest
                    if cm:
                        comprobante = cm.group(1)
                        desc = cm.group(2)

                    deb = parse_money(debs) or 0.0
                    cre = parse_money(cres) or 0.0

                    cur = {
                        "fecha_raw": fecha_raw,
                        "comprobante": comprobante,
                        "descripcion": desc.strip(),
                        "debito": float(deb),
                        "credito": float(cre),
                    }
                else:
                    # continuación de descripción
                    if cur and left:
                        cur["descripcion"] = (cur["descripcion"] + " " + left).strip()

            if cur:
                rows.append(cur)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, saldo_ini, saldo_fin, saldo_fin_fecha

    # normalizar tipos
    df["debito"] = pd.to_numeric(df["debito"], errors="coerce").fillna(0.0)
    df["credito"] = pd.to_numeric(df["credito"], errors="coerce").fillna(0.0)

    # fecha en ISO
    def _to_date(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        s = s.strip()
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                pass
        return None

    df["fecha"] = df["fecha_raw"].map(_to_date)
    return df, saldo_ini, saldo_fin, saldo_fin_fecha


# ------------------------------
# OCR fallback (solo si hace falta)
# ------------------------------
@dataclass
class OcrWord:
    text: str
    x0: int
    x1: int
    y0: int
    y1: int
    conf: int

def _render_pdf_page_to_pil(pdf_bytes: bytes, page_index: int, scale: float = 2.0) -> Image.Image:
    if pdfium is None:
        raise RuntimeError("pypdfium2 no está disponible.")
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf.get_page(page_index)
    # scale: 2.0 ~ 144 DPI; 2.5 ~ 180; 3.0 ~ 216 (aprox)
    pil = page.render(scale=scale).to_pil()
    page.close()
    pdf.close()
    return pil

def _ocr_words(img: Image.Image) -> List[OcrWord]:
    if pytesseract is None:
        raise RuntimeError("pytesseract no está disponible.")
    data = pytesseract.image_to_data(img, lang="spa", config="--psm 6", output_type=pytesseract.Output.DICT)
    out: List[OcrWord] = []
    n = len(data["text"])
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        conf = int(float(data["conf"][i])) if str(data["conf"][i]).strip() != "" else -1
        x0 = int(data["left"][i]); y0 = int(data["top"][i])
        w = int(data["width"][i]); h = int(data["height"][i])
        out.append(OcrWord(text=t, x0=x0, x1=x0+w, y0=y0, y1=y0+h, conf=conf))
    return out

def _group_ocr_to_lines(words: List[OcrWord], y_tol: int = 8) -> List[List[OcrWord]]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w.y0 / y_tol), w.x0))
    lines: List[List[OcrWord]] = []
    cur: List[OcrWord] = []
    band = None
    for w in words:
        b = round(w.y0 / y_tol)
        if band is None or b == band:
            cur.append(w); band = b
        else:
            lines.append(sorted(cur, key=lambda x: x.x0))
            cur = [w]; band = b
    if cur:
        lines.append(sorted(cur, key=lambda x: x.x0))
    return lines

def _line_text_ocr(line: List[OcrWord]) -> str:
    s = " ".join(w.text for w in line)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _detect_layout_ocr(lines: List[List[OcrWord]], img_width: int) -> Tuple[Optional[int], Optional[ColLayout]]:
    header_idx = None
    header_line = None
    for i, line in enumerate(lines):
        u = norm_text(_line_text_ocr(line))
        # tolerar "DEBITO" con o sin acento
        if all(k in u for k in ("FECHA", "DEBITO", "CREDITO", "SALDO")):
            header_idx = i
            header_line = line
            break
    if header_line is None:
        return None, None

    # centers por palabra
    def center_of(label: str) -> Optional[float]:
        lab = norm_text(label)
        for w in header_line:
            if norm_text(w.text) == lab:
                return (w.x0 + w.x1) / 2
        return None

    c_desc = center_of("DESCRIPCION")
    c_deb = center_of("DEBITO")
    c_cre = center_of("CREDITO")
    c_sal = center_of("SALDO")

    if any(v is None for v in (c_desc, c_deb, c_cre, c_sal)):
        return header_idx, None

    b_desc_deb = (c_desc + c_deb) / 2
    b_deb_cre = (c_deb + c_cre) / 2
    b_cre_sal = (c_cre + c_sal) / 2
    ok = 0 < b_desc_deb < b_deb_cre < b_cre_sal < img_width
    return header_idx, ColLayout(b_desc_deb=b_desc_deb, b_deb_cre=b_deb_cre, b_cre_sal=b_cre_sal, ok=ok)

def parse_pdf_ocr(pdf_bytes: bytes, max_pages: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[str]]:
    if pytesseract is None or pdfium is None:
        return pd.DataFrame(), None, None, None

    rows: List[dict] = []
    saldo_ini = None
    saldo_fin = None
    saldo_fin_fecha = None

    pdf = pdfium.PdfDocument(pdf_bytes)
    n_pages = len(pdf)
    pdf.close()

    if max_pages is None:
        max_pages = n_pages

    for pi in range(min(n_pages, max_pages)):
        img = _render_pdf_page_to_pil(pdf_bytes, pi, scale=2.5)
        words = _ocr_words(img)
        lines = _group_ocr_to_lines(words, y_tol=10)
        header_idx, layout = _detect_layout_ocr(lines, img.width)
        if header_idx is None or layout is None or not layout.ok:
            continue

        cur = None
        for line in lines[header_idx + 1 :]:
            # dividir palabras por columna según x center
            left_words = [w for w in line if (w.x0 + w.x1) / 2 < layout.b_desc_deb]
            deb_words = [w for w in line if layout.b_desc_deb <= (w.x0 + w.x1) / 2 < layout.b_deb_cre]
            cre_words = [w for w in line if layout.b_deb_cre <= (w.x0 + w.x1) / 2 < layout.b_cre_sal]
            sal_words = [w for w in line if (w.x0 + w.x1) / 2 >= layout.b_cre_sal]

            left = _line_text_ocr(left_words)
            debs = _line_text_ocr(deb_words)
            cres = _line_text_ocr(cre_words)
            sals = _line_text_ocr(sal_words)

            if not (left or debs or cres or sals):
                continue

            u = norm_text(left)

            if "SALDO ANTERIOR" in u:
                m = parse_money(sals) or parse_money(cres) or parse_money(debs) or parse_money(left)
                if m is not None:
                    saldo_ini = m
                continue

            if "SALDO AL" in u:
                m = parse_money(sals) or parse_money(cres) or parse_money(debs) or parse_money(left)
                if m is not None:
                    saldo_fin = m
                dm = DATE_ANY_RE.search(left)
                if dm:
                    saldo_fin_fecha = dm.group(1)
                continue

            dm = DATE_AT_START_RE.search(left)
            if dm:
                if cur:
                    rows.append(cur)

                fecha_raw = dm.group(1)
                rest = left[dm.end() :].strip()
                cm = re.match(r"^(\d+)\s+(.*)$", rest)
                comprobante = None
                desc = rest
                if cm:
                    comprobante = cm.group(1)
                    desc = cm.group(2)

                deb = parse_money(debs) or 0.0
                cre = parse_money(cres) or 0.0

                cur = {
                    "fecha_raw": fecha_raw,
                    "comprobante": comprobante,
                    "descripcion": desc.strip(),
                    "debito": float(deb),
                    "credito": float(cre),
                }
            else:
                if cur and left:
                    cur["descripcion"] = (cur["descripcion"] + " " + left).strip()

        if cur:
            rows.append(cur)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, saldo_ini, saldo_fin, saldo_fin_fecha

    df["debito"] = pd.to_numeric(df["debito"], errors="coerce").fillna(0.0)
    df["credito"] = pd.to_numeric(df["credito"], errors="coerce").fillna(0.0)

    def _to_date(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        s = s.strip()
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                pass
        return None

    df["fecha"] = df["fecha_raw"].map(_to_date)
    return df, saldo_ini, saldo_fin, saldo_fin_fecha


# ------------------------------
# Clasificación: Resumen Operativo / Préstamos
# ------------------------------
def classify_operativo(desc: str) -> Optional[str]:
    u = norm_text(desc)
    if not u:
        return None

    # Percepciones IVA
    if ("PERCEP" in u or "PERCEPC" in u) and "IVA" in u:
        return "Percepciones de IVA"

    # Ley 25.413
    if "25.413" in u or "25413" in u:
        return "Impuesto Ley 25.413"

    # SIRCREB / IIBB
    if "SIRCREB" in u:
        return "SIRCREB"

    # IVA (no percepciones)
    if "IVA" in u and "PERCEP" not in u and "PERCEPC" not in u:
        if "10,5" in u or "10.5" in u or "10,50" in u:
            return "IVA 10.5"
        return "IVA"

    # Comisiones / Gastos bancarios neto
    if any(k in u for k in ("COMISION", "COMIS.", "COMISIO", "GASTOS BANCARIOS", "MANTENIMIENTO", "PAQUETE", "SERVICIO")):
        if "10,5" in u or "10.5" in u or "10,50" in u:
            return "Gastos Bancarios 10.5"
        return "Comisiones (Gastos Bancarios) Neto"

    # Exentos (sellos, etc.) — mantener acotado
    if any(k in u for k in ("SELLO", "SELLADO", "EXENTO")):
        return "Gastos Exentos"

    return None

def is_loan(desc: str) -> bool:
    u = norm_text(desc)
    return any(k in u for k in ("PRESTAMO", "PRÉSTAMO", "CUOTA PREST", "CUOTA PRESTAM", "ACRED PREST", "CRED PREST"))

def build_resumen_operativo(df_mov: pd.DataFrame) -> pd.DataFrame:
    if df_mov.empty:
        return pd.DataFrame(columns=["Concepto", "Débitos", "Créditos", "Total (Débitos - Créditos)"])

    tmp = df_mov.copy()
    tmp["Concepto"] = tmp["descripcion"].map(classify_operativo)
    tmp = tmp[tmp["Concepto"].notna()].copy()

    if tmp.empty:
        # devolver estructura vacía con filas esperadas
        base = pd.DataFrame({"Concepto": [
            "Comisiones (Gastos Bancarios) Neto",
            "IVA",
            "Gastos Bancarios 10.5",
            "IVA 10.5",
            "Impuesto Ley 25.413",
            "Percepciones de IVA",
            "SIRCREB",
            "Gastos Exentos",
        ]})
        base["Débitos"] = 0.0
        base["Créditos"] = 0.0
        base["Total (Débitos - Créditos)"] = 0.0
        total = pd.DataFrame([{"Concepto":"Total","Débitos":0.0,"Créditos":0.0,"Total (Débitos - Créditos)":0.0}])
        return pd.concat([base, total], ignore_index=True)

    grp = tmp.groupby("Concepto", as_index=False).agg({"debito":"sum","credito":"sum"})
    grp["Total (Débitos - Créditos)"] = grp["debito"] - grp["credito"]
    grp = grp.rename(columns={"debito":"Débitos","credito":"Créditos"})

    order = [
        "Comisiones (Gastos Bancarios) Neto",
        "IVA",
        "Gastos Bancarios 10.5",
        "IVA 10.5",
        "Impuesto Ley 25.413",
        "Percepciones de IVA",
        "SIRCREB",
        "Gastos Exentos",
    ]
    grp = grp.set_index("Concepto").reindex(order).fillna(0.0).reset_index()

    total = {
        "Concepto": "Total",
        "Débitos": float(grp["Débitos"].sum()),
        "Créditos": float(grp["Créditos"].sum()),
        "Total (Débitos - Créditos)": float(grp["Total (Débitos - Créditos)"].sum()),
    }
    grp = pd.concat([grp, pd.DataFrame([total])], ignore_index=True)
    return grp

def build_prestamos(df_mov: pd.DataFrame) -> pd.DataFrame:
    if df_mov.empty:
        return pd.DataFrame()
    tmp = df_mov.copy()
    tmp = tmp[tmp["descripcion"].map(is_loan)].copy()
    if tmp.empty:
        return tmp
    tmp["tipo"] = np.where(tmp["credito"] > 0, "Acreditación Préstamos", "Cuota de préstamo")
    cols = ["fecha_raw", "fecha", "comprobante", "descripcion", "debito", "credito", "tipo"]
    return tmp[cols].reset_index(drop=True)

def build_creditos(df_mov: pd.DataFrame) -> pd.DataFrame:
    if df_mov.empty:
        return pd.DataFrame()
    tmp = df_mov.copy()
    tmp = tmp[(tmp["credito"] > 0) & (~tmp["descripcion"].map(is_loan))].copy()
    cols = ["fecha_raw", "fecha", "comprobante", "descripcion", "credito"]
    return tmp[cols].reset_index(drop=True)


# ------------------------------
# Export: Excel + PDF
# ------------------------------
def make_excel(df_mov: pd.DataFrame, df_oper: pd.DataFrame, df_prest: pd.DataFrame, df_cred: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        workbook = writer.book
        fmt_num = workbook.add_format({"num_format": "#,##0.00"})
        fmt_hdr = workbook.add_format({"bold": True, "bg_color": "#F3F4F6"})
        fmt_txt = workbook.add_format({"text_wrap": True})

        def write_df(sheet: str, df: pd.DataFrame):
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            # header format
            for col, name in enumerate(df.columns):
                ws.write(0, col, name, fmt_hdr)
            # col widths
            for col, name in enumerate(df.columns):
                width = min(60, max(10, int(df[name].astype(str).map(len).quantile(0.9)) if not df.empty else 10))
                ws.set_column(col, col, width, fmt_txt if name.lower().startswith("descrip") else None)
            # numeric columns
            for col, name in enumerate(df.columns):
                if name.lower() in ("debito","credito","débitos","créditos","total (débitos - créditos)","total"):
                    ws.set_column(col, col, 16, fmt_num)

        write_df("Movimientos", df_mov)
        write_df("Resumen_Operativo", df_oper)
        if not df_prest.empty:
            write_df("Prestamos", df_prest)
        if not df_cred.empty:
            write_df("Creditos", df_cred)

    bio.seek(0)
    return bio.getvalue()

def make_pdf_resumen_operativo(df_oper: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("Resumen Operativo (Banco Credicoop)", styles["Title"]))
    elems.append(Spacer(1, 10))

    # Tabla
    data = [list(df_oper.columns)] + df_oper.values.tolist()
    # formatear num en tabla
    for r in range(1, len(data)):
        for c in range(1, len(data[0])):
            try:
                data[r][c] = fmt_ar(float(data[r][c]))
            except Exception:
                pass

    table = Table(data, colWidths=[200, 90, 90, 110])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111827")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D1D5DB")),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, colors.HexColor("#FAFAFA")]),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#FFF7ED")),
        ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
    ]))
    elems.append(table)

    doc.build(elems)
    bio.seek(0)
    return bio.getvalue()


# ------------------------------
# Orquestación
# ------------------------------
@st.cache_data(show_spinner=False)
def parse_pdf(pdf_bytes: bytes) -> Dict[str, object]:
    # 1) Intentar CHARS
    df, s_ini, s_fin, s_fin_fecha = parse_pdf_chars(pdf_bytes)

    # aceptar chars si trae suficientes señales
    ok_chars = (not df.empty) and (s_ini is not None) and (s_fin is not None)
    if ok_chars:
        mode = "CHARS"
    else:
        # 2) OCR fallback
        df2, si2, sf2, sff2 = parse_pdf_ocr(pdf_bytes, max_pages=None)
        if df2.empty:
            # devolver lo que haya (chars) para debugging
            mode = "CHARS"
        else:
            df, s_ini, s_fin, s_fin_fecha = df2, si2, sf2, sff2
            mode = "OCR"

    # post-procesar
    if df.empty:
        return {
            "mode": mode,
            "df": df,
            "saldo_ini": s_ini,
            "saldo_fin": s_fin,
            "saldo_fin_fecha": s_fin_fecha,
            "oper": build_resumen_operativo(df),
            "prest": build_prestamos(df),
            "cred": build_creditos(df),
        }

    df = df.copy()
    df["tipo"] = np.where(df["debito"] > 0, "Débito", np.where(df["credito"] > 0, "Crédito", ""))
    df["categoria_operativo"] = df["descripcion"].map(classify_operativo).fillna("")

    oper = build_resumen_operativo(df)
    prest = build_prestamos(df)
    cred = build_creditos(df)

    return {
        "mode": mode,
        "df": df,
        "saldo_ini": s_ini,
        "saldo_fin": s_fin,
        "saldo_fin_fecha": s_fin_fecha,
        "oper": oper,
        "prest": prest,
        "cred": cred,
    }


# ------------------------------
# Streamlit App
# ------------------------------
def run_app():
    st.set_page_config(
        page_title="IA Resumen Credicoop",
        page_icon=str(FAVICON) if FAVICON.exists() else None,
        layout="centered",
    )

    # estilos suaves
    st.markdown(
        """
        <style>
        .block-container {max-width: 980px; padding-top: 1.2rem; padding-bottom: 2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if LOGO.exists():
        st.image(str(LOGO), width=180)

    st.title("IA Resumen Credicoop")
    st.caption("Parser para Resumen de Cuenta Corriente Comercial (PDF) — Banco Credicoop")

    up = st.file_uploader("Subí un PDF (un solo archivo)", type=["pdf"], accept_multiple_files=False)

    if not up:
        st.info("Subí el PDF para procesarlo.")
        return

    pdf_bytes = up.read()

    with st.spinner("Procesando PDF…"):
        out = parse_pdf(pdf_bytes)

    df: pd.DataFrame = out["df"]
    saldo_ini = out["saldo_ini"]
    saldo_fin = out["saldo_fin"]
    saldo_fin_fecha = out["saldo_fin_fecha"]
    mode = out["mode"]

    if df.empty:
        st.error("No se pudieron extraer movimientos. Revisá que el PDF sea un resumen Credicoop con tabla FECHA/COMBTE/DESCRIPCION/DEBITO/CREDITO/SALDO.")
        st.caption(f"Modo de lectura: {mode}")
        return

    total_deb = float(df["debito"].sum())
    total_cre = float(df["credito"].sum())

    saldo_calc = None
    diff = None
    if saldo_ini is not None:
        saldo_calc = float(safe_float(saldo_ini) + total_cre - total_deb)
    if (saldo_calc is not None) and (saldo_fin is not None):
        diff = float(saldo_calc - safe_float(saldo_fin))

    # métricas (sin símbolo $)
    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo inicial", fmt_ar(saldo_ini) if saldo_ini is not None else "")
    c2.metric("Total créditos (+)", fmt_ar(total_cre))
    c3.metric("Total débitos (−)", fmt_ar(total_deb))

    c4, c5, c6 = st.columns(3)
    c4.metric("Saldo al (PDF)", fmt_ar(saldo_fin) if saldo_fin is not None else "")
    c5.metric("Saldo calculado", fmt_ar(saldo_calc) if saldo_calc is not None else "")
    c6.metric("Diferencia", fmt_ar(diff) if diff is not None else "")

    st.caption(f"Saldo final tomado de: {saldo_fin_fecha or '—'}")
    st.caption(f"Modo de lectura: {mode}")

    if diff is None or abs(diff) > 0.01:
        st.error("No cuadra la conciliación (estricto).")

    st.divider()

    # Resumen Operativo
    st.subheader("Resumen Operativo")
    df_oper: pd.DataFrame = out["oper"].copy()
    df_show = df_oper.copy()
    for col in df_show.columns[1:]:
        df_show[col] = df_show[col].map(fmt_ar)
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Préstamos
    df_prest: pd.DataFrame = out["prest"]
    if not df_prest.empty:
        st.subheader("Préstamos (detectados)")
        dfp = df_prest.copy()
        dfp["debito"] = dfp["debito"].map(fmt_ar)
        dfp["credito"] = dfp["credito"].map(fmt_ar)
        st.dataframe(dfp, use_container_width=True, hide_index=True)

    # Créditos
    df_cred: pd.DataFrame = out["cred"]
    if not df_cred.empty:
        st.subheader("Créditos (no préstamo)")
        dfc = df_cred.copy()
        dfc["credito"] = dfc["credito"].map(fmt_ar)
        st.dataframe(dfc, use_container_width=True, hide_index=True)

    st.subheader("Movimientos")
    dfm = df.copy()
    dfm["debito"] = dfm["debito"].map(fmt_ar)
    dfm["credito"] = dfm["credito"].map(fmt_ar)
    st.dataframe(dfm[["fecha_raw","comprobante","descripcion","categoria_operativo","tipo","debito","credito"]], use_container_width=True, hide_index=True)

    # Descargas
    st.divider()
    st.subheader("Descargas")

    excel_bytes = make_excel(df, df_oper, df_prest, df_cred)
    pdf_bytes_out = make_pdf_resumen_operativo(df_oper)

    st.download_button(
        "Descargar Excel",
        data=excel_bytes,
        file_name="credicoop_resumen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.download_button(
        "Descargar PDF (Resumen Operativo)",
        data=pdf_bytes_out,
        file_name="credicoop_resumen_operativo.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.caption("© AIE San Justo — Herramienta para uso interno")


if __name__ == "__main__":
    run_app()