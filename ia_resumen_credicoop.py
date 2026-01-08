# ia_resumen_credicoop.py
# IA Resumen Credicoop (Cuenta Corriente Comercial PDF)
# Herramienta para uso interno - AIE San Justo
#
# Parser híbrido:
#  - CHARS (pdfplumber.page.chars): para PDFs con texto seleccionable aunque esté "letter-spaced"
#  - OCR (Tesseract): fallback para PDFs CID/Type3 o escaneados
#
# Reglas Credicoop:
#  - Un movimiento comienza en una línea que tiene FECHA (dd/mm/aa o dd/mm/aaaa).
#  - Si una descripción ocupa 2 renglones y sólo el primero tiene fecha, el segundo es continuación.
#  - Nunca hay débito y crédito juntos en un movimiento (si OCR mete ambos, se preserva y se ve en grilla).
#  - Si hay 2 montos en la misma línea, el de la derecha es SALDO diario (NO se usa).
#  - Conciliación estricta: Saldo anterior + Créditos - Débitos = Saldo AL (fecha).
#  - Débito siempre a la izquierda, Crédito a la derecha.
#  - Saldos pueden ser negativos.
#
# Developer: Alfonso Alderete

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- deps diferidas ----------------
try:
    import pdfplumber
except Exception as e:
    st.error(f"No se pudo importar pdfplumber: {e}\nRevisá requirements.txt")
    st.stop()

try:
    import pytesseract
    from PIL import Image
except Exception as e:
    st.error(f"No se pudo importar pytesseract/Pillow: {e}\nRevisá requirements.txt")
    st.stop()

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ---------------- UI / assets ----------------
HERE = Path(__file__).parent
LOGO = HERE / "logo_aie.png"
FAVICON = HERE / "favicon-aie.ico"

st.set_page_config(
    page_title="IA Resumen Credicoop",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="centered",
)

if LOGO.exists():
    st.image(str(LOGO), width=200)

st.title("IA Resumen Credicoop")

st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- regex ----------------
DATE_RE = re.compile(r"^\s*(\d{2})[\/⁄](\d{2})[\/⁄](\d{2}|\d{4})\b")
DATE_ANY_RE = re.compile(r"\b\d{2}[\/⁄]\d{2}[\/⁄]\d{2,4}\b")

MONEY_STRICT_RE = re.compile(r"^-?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2}$")
MONEY_FUZZY_RE = re.compile(r"-?\d{1,3}(?:[.\s']?\d{3})*,\d{2}")

LONG_INT_RE = re.compile(r"\b\d{6,}\b")


def fmt_ar(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    return f"{n:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


def normalize_money(tok: str) -> float:
    if tok is None:
        return np.nan
    s = str(tok).strip()
    s = s.replace("−", "-").replace(" ", "")
    s = s.replace("'", "")
    neg = s.startswith("-")
    s2 = s.lstrip("-")
    if "," not in s2:
        return np.nan
    main, frac = s2.rsplit(",", 1)
    main = main.replace(".", "")
    frac = re.sub(r"\D", "", frac)[:2]
    if len(frac) != 2:
        return np.nan
    try:
        v = float(f"{main}.{frac}")
        return -v if neg else v
    except Exception:
        return np.nan


def norm_txt(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    s = s.replace("⁄", "/")
    return re.sub(r"\s{2,}", " ", s)


def normalize_desc(desc: str) -> str:
    u = (desc or "").upper()
    u = u.replace(".", "")  # I.V.A. => IVA
    u = LONG_INT_RE.sub("", u)
    u = " ".join(u.split())
    return u


def is_cid_pdf(pdf) -> bool:
    try:
        t = pdf.pages[0].extract_text() or ""
        return "(cid:" in t
    except Exception:
        return False


# ---------------- clasificación ----------------
def clasificar(desc: str) -> str:
    n = normalize_desc(desc)

    if "25413" in n or "25.413" in n or "LEY 25413" in n or "IMPUESTO LEY 25413" in n or "IMPUESTO A LOS DEBITOS Y CREDITOS" in n:
        return "Ley 25.413"

    if "SIRCREB" in n:
        return "SIRCREB"

    if ("PERCEP" in n and "IVA" in n) or ("RG 2408" in n) or ("RG2408" in n) or ("2408" in n and "IVA" in n):
        return "Percepciones de IVA"

    if "DEBITO FISCAL" in n and "IVA" in n:
        if "10,5" in n or "10.5" in n or "10 5" in n:
            return "IVA 10,5%"
        return "IVA 21%"

    if ("INTERES" in n or "SALDO DEUDOR" in n or "INT " in n) and "IVA" not in n:
        return "Comisiones/Gastos Neto 10,5%"

    # comisiones/gastos neto 21 por defecto (evitando IVA y ley)
    if any(k in n for k in ["COMISION", "SERVICIO", "MANTEN", "GASTO", "CARGO", "PAQUETE"]) and "IVA" not in n and "DEBITO FISCAL" not in n:
        return "Comisiones/Gastos Neto 21%"

    if re.search(r"\bPREST", n):
        return "Préstamos"

    return "Otros"


# ---------------- OCR helpers ----------------
@dataclass
class Tok:
    text: str
    x0: float
    x1: float
    top: float


def _group_lines(tokens: List[Tok], ytol: float = 9.0) -> List[List[Tok]]:
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t.top, t.x0))
    lines: List[List[Tok]] = []
    cur: List[Tok] = []
    y0 = None
    for t in toks:
        if y0 is None or abs(t.top - y0) <= ytol:
            cur.append(t)
            if y0 is None:
                y0 = t.top
        else:
            lines.append(sorted(cur, key=lambda z: z.x0))
            cur = [t]
            y0 = t.top
    if cur:
        lines.append(sorted(cur, key=lambda z: z.x0))
    return lines


def _money_tokens_from_line(line: List[Tok]) -> List[Tuple[float, float, str, float]]:
    """
    Extrae montos del renglón. Re-arma casos OCR partidos:
      "1.388," + "05" => "1.388,05"
    Devuelve lista de (x0,x1,text,val).
    """
    out = []
    i = 0
    while i < len(line):
        t = norm_txt(line[i].text)
        if not t:
            i += 1
            continue

        # caso partido (termina con coma + siguiente token de 2 dígitos)
        if (t.endswith(",") or ("," in t and not re.search(r",\d{2}$", t))) and i + 1 < len(line):
            nxt = norm_txt(line[i + 1].text)
            if re.fullmatch(r"\d{2}", nxt):
                cand = (t + nxt).replace(" ", "")
                if MONEY_STRICT_RE.match(cand) or MONEY_FUZZY_RE.fullmatch(cand):
                    val = normalize_money(cand)
                    if not np.isnan(val):
                        out.append((line[i].x0, line[i + 1].x1, cand, float(val)))
                        i += 2
                        continue

        t2 = t.replace(" ", "")
        if MONEY_STRICT_RE.match(t2) or MONEY_FUZZY_RE.fullmatch(t2):
            val = normalize_money(t2)
            if not np.isnan(val):
                out.append((line[i].x0, line[i].x1, t2, float(val)))

        i += 1
    return out


def _line_text(line: List[Tok]) -> str:
    return " ".join(norm_txt(t.text) for t in line if norm_txt(t.text))


def _detect_date_at_start(line: List[Tok]) -> Optional[str]:
    if not line:
        return None
    full = _line_text(line)
    m = DATE_RE.match(full)
    if m:
        return m.group(0).replace("⁄", "/")
    return None


def _parse_date(s: str) -> Optional[datetime]:
    s = norm_txt(s).replace("⁄", "/")
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _ocr_page_tokens(page, resolution: int = 240, crop_top: int = 40, crop_bottom: int = 40) -> Tuple[List[Tok], int, int]:
    """
    OCR de página con recorte suave (para no cortar SALDO ANTERIOR).
    """
    crop = page.crop((0, crop_top, page.width, page.height - crop_bottom))
    img = crop.to_image(resolution=resolution).original.convert("RGB")
    cfg = "--psm 6"
    df = pytesseract.image_to_data(img, lang="spa", config=cfg, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).map(lambda s: s.strip())
    df = df[df["text"] != ""]
    toks = [Tok(text=str(r.text), x0=float(r.left), x1=float(r.left + r.width), top=float(r.top))
            for r in df.itertuples(index=False)]
    return toks, img.width, img.height


def _kmeans_1d_two_clusters(xs: List[float], iters: int = 12) -> Optional[Tuple[float, float, float]]:
    xs = [float(x) for x in xs if x is not None and not np.isnan(x)]
    if len(xs) < 6:
        return None
    arr = np.array(xs, dtype=float)
    c1 = np.quantile(arr, 0.33)
    c2 = np.quantile(arr, 0.66)
    for _ in range(iters):
        d1 = np.abs(arr - c1)
        d2 = np.abs(arr - c2)
        lab = d1 <= d2
        if lab.all() or (~lab).all():
            break
        c1n = arr[lab].mean()
        c2n = arr[~lab].mean()
        c1, c2 = c1n, c2n
    left, right = (c1, c2) if c1 < c2 else (c2, c1)
    thr = (left + right) / 2.0
    return float(left), float(right), float(thr)


# ---------------- OCR parsing ----------------
def parse_ocr(pdf_bytes: bytes, pdf_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    saldo_inicial = np.nan
    saldo_final = np.nan
    fecha_final = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            toks, w, _ = _ocr_page_tokens(page, resolution=240, crop_top=40, crop_bottom=40)
            lines = _group_lines(toks, ytol=9.0)

            # anclas de saldo
            for ln in lines:
                line_txt = normalize_desc(_line_text(ln))
                mny = _money_tokens_from_line(ln)
                if not mny:
                    continue
                mny_sorted = sorted(mny, key=lambda x: x[1])
                if "SALDO ANTERIOR" in line_txt:
                    saldo_inicial = float(mny_sorted[-1][3])
                if "SALDO AL" in line_txt:
                    md = DATE_ANY_RE.search(_line_text(ln))
                    if md:
                        fecha_final = md.group(0).replace("⁄", "/")
                    saldo_final = float(mny_sorted[-1][3])

            current = None
            x_centers = []
            items = []

            for ln in lines:
                date_raw = _detect_date_at_start(ln)
                mny = _money_tokens_from_line(ln)

                if date_raw:
                    if current is not None:
                        items.append(current)

                    current = {
                        "fecha_raw": date_raw,
                        "comprobante": "",
                        "descripcion": "",
                        "amounts": [],      # (xc,val)
                        "pagina": pageno,
                    }

                    # comprobante: primer token numérico 3+ dígitos
                    for t in ln:
                        tt = norm_txt(t.text)
                        if tt.isdigit() and len(tt) >= 3:
                            current["comprobante"] = tt
                            break

                    # descripción: tokens no monetarios
                    parts = []
                    for t in ln:
                        tt = norm_txt(t.text)
                        if not tt:
                            continue
                        if DATE_ANY_RE.fullmatch(tt):
                            continue
                        if current["comprobante"] and tt == current["comprobante"]:
                            continue
                        if MONEY_STRICT_RE.match(tt.replace(" ", "")) or MONEY_FUZZY_RE.fullmatch(tt.replace(" ", "")) or tt.endswith(","):
                            continue
                        parts.append(tt)

                    desc = " ".join(parts).strip()
                    if desc.startswith(date_raw):
                        desc = desc[len(date_raw):].strip()
                    current["descripcion"] = desc

                    # importes: si hay 2+, descartar el último (saldo diario)
                    if mny:
                        mny_sorted = sorted(mny, key=lambda x: x[1])
                        if len(mny_sorted) >= 2:
                            mny_sorted = mny_sorted[:-1]
                        for x0, x1, _, val in mny_sorted:
                            xc = (x0 + x1) / 2.0
                            current["amounts"].append((xc, float(val)))
                            x_centers.append(xc)

                else:
                    if current is None:
                        continue

                    # continuación descripción
                    parts = []
                    for t in ln:
                        tt = norm_txt(t.text)
                        if not tt:
                            continue
                        if MONEY_STRICT_RE.match(tt.replace(" ", "")) or MONEY_FUZZY_RE.fullmatch(tt.replace(" ", "")) or tt.endswith(","):
                            continue
                        if DATE_ANY_RE.search(tt):
                            continue
                        parts.append(tt)
                    extra = " ".join(parts).strip()
                    if extra:
                        current["descripcion"] = (current["descripcion"] + " " + extra).strip()

                    # si trae importes, anexar al movimiento (sin inventar fecha nueva)
                    if mny:
                        mny_sorted = sorted(mny, key=lambda x: x[1])
                        if len(mny_sorted) >= 2:
                            mny_sorted = mny_sorted[:-1]
                        for x0, x1, _, val in mny_sorted:
                            xc = (x0 + x1) / 2.0
                            current["amounts"].append((xc, float(val)))
                            x_centers.append(xc)

            if current is not None:
                items.append(current)

            km = _kmeans_1d_two_clusters(x_centers)
            thr = km[2] if km else (w * 0.68)

            for it in items:
                dt = _parse_date(it["fecha_raw"])
                if dt is None:
                    continue

                deb = 0.0
                cre = 0.0
                for xc, val in it["amounts"]:
                    if xc <= thr:
                        deb += float(val)
                    else:
                        cre += float(val)

                desc = norm_txt(it["descripcion"])
                rows.append({
                    "fecha": dt.date(),
                    "fecha_raw": it["fecha_raw"],
                    "comprobante": it["comprobante"] or "",
                    "descripcion": desc,
                    "desc_norm": normalize_desc(desc),
                    "debito": float(deb) if deb else 0.0,
                    "credito": float(cre) if cre else 0.0,
                    "Clasificación": clasificar(desc),
                    "archivo": pdf_name,
                    "pagina": it["pagina"],
                })

    meta = {
        "saldo_inicial": float(saldo_inicial) if not np.isnan(saldo_inicial) else np.nan,
        "saldo_final": float(saldo_final) if not np.isnan(saldo_final) else np.nan,
        "fecha_final": fecha_final,
    }
    return pd.DataFrame(rows), meta


# ---------------- CHARS parsing ----------------
def _chars_lines(page, ytol: float = 2.2):
    chars = page.chars or []
    chars_sorted = sorted(chars, key=lambda c: (round(float(c.get("top", 0)) / ytol), float(c.get("x0", 0))))
    lines = []
    cur = []
    band = None
    for c in chars_sorted:
        b = round(float(c.get("top", 0)) / ytol)
        if band is None or b == band:
            cur.append(c)
        else:
            lines.append(cur)
            cur = [c]
        band = b
    if cur:
        lines.append(cur)

    out_lines = []
    for ln in lines:
        ln = sorted(ln, key=lambda c: float(c.get("x0", 0)))
        text = "".join(c.get("text", "") for c in ln)
        text = norm_txt(text)
        if text:
            out_lines.append(text)
    return out_lines


def _money_spans_from_text(text: str) -> List[str]:
    toks = []
    for token in re.split(r"\s+", text):
        t = token.strip().replace("−", "-")
        if not t:
            continue
        if MONEY_STRICT_RE.match(t) or MONEY_FUZZY_RE.fullmatch(t):
            toks.append(t)
    return toks


def parse_chars(pdf_bytes: bytes, pdf_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    saldo_inicial = np.nan
    saldo_final = np.nan
    fecha_final = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            lines = _chars_lines(page, ytol=2.2)

            for txt in lines:
                up = normalize_desc(txt)
                ms = _money_spans_from_text(txt)
                if not ms:
                    continue
                last_val = normalize_money(ms[-1])
                if np.isnan(last_val):
                    continue
                if "SALDO ANTERIOR" in up:
                    saldo_inicial = float(last_val)
                if "SALDO AL" in up:
                    md = DATE_ANY_RE.search(txt)
                    if md:
                        fecha_final = md.group(0).replace("⁄", "/")
                    saldo_final = float(last_val)

            current = None
            x_centers = []
            items = []

            for txt in lines:
                date_m = DATE_RE.match(txt)
                ms = _money_spans_from_text(txt)

                if date_m:
                    if current is not None:
                        items.append(current)
                    date_raw = date_m.group(0).replace("⁄", "/")
                    current = {"fecha_raw": date_raw, "comprobante": "", "descripcion": "", "amounts": [], "pagina": pageno}

                    mcomp = re.search(r"\b\d{3,}\b", txt)
                    if mcomp:
                        current["comprobante"] = mcomp.group(0)

                    desc = txt[len(date_raw):].strip()
                    if ms:
                        # cortar en el primer monto
                        for m in ms:
                            pos = txt.find(m)
                            if pos != -1:
                                desc = txt[len(date_raw):pos].strip()
                                break
                    current["descripcion"] = desc

                    if ms:
                        mm = ms[:-1] if len(ms) >= 2 else ms
                        for m in mm:
                            xc = float(txt.find(m))
                            val = normalize_money(m)
                            if not np.isnan(val):
                                current["amounts"].append((xc, float(val)))
                                x_centers.append(xc)
                else:
                    if current is None:
                        continue
                    # concatenar continuación
                    if txt and not DATE_ANY_RE.search(txt):
                        current["descripcion"] = (current["descripcion"] + " " + txt).strip()

                    if ms:
                        mm = ms[:-1] if len(ms) >= 2 else ms
                        for m in mm:
                            xc = float(txt.find(m))
                            val = normalize_money(m)
                            if not np.isnan(val):
                                current["amounts"].append((xc, float(val)))
                                x_centers.append(xc)

            if current is not None:
                items.append(current)

            km = _kmeans_1d_two_clusters(x_centers)
            thr = km[2] if km else 140.0

            for it in items:
                dt = _parse_date(it["fecha_raw"])
                if dt is None:
                    continue
                deb = 0.0
                cre = 0.0
                for xc, val in it["amounts"]:
                    if xc <= thr:
                        deb += float(val)
                    else:
                        cre += float(val)

                desc = norm_txt(it["descripcion"])
                rows.append({
                    "fecha": dt.date(),
                    "fecha_raw": it["fecha_raw"],
                    "comprobante": it["comprobante"] or "",
                    "descripcion": desc,
                    "desc_norm": normalize_desc(desc),
                    "debito": float(deb) if deb else 0.0,
                    "credito": float(cre) if cre else 0.0,
                    "Clasificación": clasificar(desc),
                    "archivo": pdf_name,
                    "pagina": it["pagina"],
                })

    meta = {
        "saldo_inicial": float(saldo_inicial) if not np.isnan(saldo_inicial) else np.nan,
        "saldo_final": float(saldo_final) if not np.isnan(saldo_final) else np.nan,
        "fecha_final": fecha_final,
    }
    return pd.DataFrame(rows), meta


# ---------------- Selector ----------------
def parse_pdf(pdf_bytes: bytes, pdf_name: str) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        cid = is_cid_pdf(pdf)

    df_chars = pd.DataFrame()
    meta_chars = {}
    if not cid:
        df_chars, meta_chars = parse_chars(pdf_bytes, pdf_name)

    df_ocr, meta_ocr = parse_ocr(pdf_bytes, pdf_name)

    def diff(df, meta):
        si = meta.get("saldo_inicial", np.nan)
        sf = meta.get("saldo_final", np.nan)
        if df is None or df.empty or np.isnan(si) or np.isnan(sf):
            return np.inf
        td = float(df["debito"].sum())
        tc = float(df["credito"].sum())
        calc = float(si) + tc - td
        return abs(calc - float(sf))

    d_chars = diff(df_chars, meta_chars) if not df_chars.empty else np.inf
    d_ocr = diff(df_ocr, meta_ocr) if not df_ocr.empty else np.inf

    if d_chars <= d_ocr:
        return df_chars, meta_chars, "CHARS"
    return df_ocr, meta_ocr, "OCR"


# ---------------- Resumen Operativo (Concepto / Importe) ----------------
def build_resumen_operativo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Concepto", "Importe"])

    def deb_sum(mask):
        return float(df.loc[mask, "debito"].sum())

    def cred_sum(mask):
        return float(df.loc[mask, "credito"].sum())

    m_neto21 = df["Clasificación"].eq("Comisiones/Gastos Neto 21%")
    m_neto105 = df["Clasificación"].eq("Comisiones/Gastos Neto 10,5%")
    m_iva21 = df["Clasificación"].eq("IVA 21%")
    m_iva105 = df["Clasificación"].eq("IVA 10,5%")
    m_piva = df["Clasificación"].eq("Percepciones de IVA")
    m_sir = df["Clasificación"].eq("SIRCREB")
    m_ley = df["Clasificación"].eq("Ley 25.413")

    neto21 = deb_sum(m_neto21)
    iva21 = deb_sum(m_iva21)
    neto105 = deb_sum(m_neto105)
    iva105 = deb_sum(m_iva105)
    percep = deb_sum(m_piva)
    sircreb = deb_sum(m_sir)

    ley_deb = deb_sum(m_ley)
    ley_cred = cred_sum(m_ley)
    ley_neto = ley_deb - ley_cred

    total = neto21 + iva21 + neto105 + iva105 + percep + sircreb + ley_neto

    rows = [
        ["Comisiones/Gastos al 21% (Neto)", neto21],
        ["IVA 21% (sobre comisiones/gastos)", iva21],
        ["Comisiones/Gastos al 10,5% (Neto)", neto105],
        ["IVA 10,5% (sobre comisiones/gastos)", iva105],
        ["Percepciones de IVA", percep],
        ["SIRCREB", sircreb],
        ["Ley 25.413 (DyC) – Neto (Débitos − Créditos)", ley_neto],
        ["TOTAL", total],
    ]
    return pd.DataFrame(rows, columns=["Concepto", "Importe"])


def build_prestamos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    m = df["Clasificación"].eq("Préstamos")
    cols = ["fecha_raw", "fecha", "comprobante", "descripcion", "debito", "credito", "archivo", "pagina"]
    return df.loc[m, cols].copy()


# ---------------- Descargas ----------------
def df_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for name, d in sheets.items():
            d.to_excel(writer, index=False, sheet_name=(name[:31] if name else "Sheet1"))

        wb = writer.book
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        date_fmt = wb.add_format({"num_format": "dd/mm/yyyy"})

        for sh_name, ws in writer.sheets.items():
            df_sheet = sheets.get(sh_name)
            if df_sheet is None:
                continue
            for j, col in enumerate(df_sheet.columns):
                width = min(max(len(str(col)), 12) + 2, 52)
                ws.set_column(j, j, width)
            for colname in ["debito", "credito", "Importe"]:
                if colname in df_sheet.columns:
                    j = list(df_sheet.columns).index(colname)
                    ws.set_column(j, j, 18, money_fmt)
            if "fecha" in df_sheet.columns:
                j = list(df_sheet.columns).index("fecha")
                ws.set_column(j, j, 14, date_fmt)

    return out.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------- PDF Resumen Operativo ----------------
def resumen_operativo_pdf_bytes(df_res: pd.DataFrame, title: str = "Resumen Operativo: Registración Módulo IVA (Credicoop)") -> Optional[bytes]:
    if not REPORTLAB_OK or df_res is None or df_res.empty:
        return None

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    elems = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 10),
    ]

    data = [["Concepto", "Importe"]] + [[str(r["Concepto"]), fmt_ar(float(r["Importe"]))] for _, r in df_res.iterrows()]
    tbl = Table(data, colWidths=[340, 160])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Herramienta para uso interno AIE San Justo | Developer Alfonso Alderete", styles["Normal"]))
    doc.build(elems)
    return pdf_buf.getvalue()


# ---------------- App ----------------
uploaded = st.file_uploader("Subí un PDF del resumen bancario (Banco Credicoop)", type=["pdf"])
if uploaded is None:
    st.info("La app no almacena datos. Procesamiento local en memoria.")
    st.stop()

data = uploaded.read()
pdf_name = uploaded.name

with st.spinner("Procesando PDF..."):
    df, meta, modo = parse_pdf(data, pdf_name)

st.caption(f"Modo de lectura: {modo}")

if df is None or df.empty:
    st.error("No se detectaron movimientos. Revisá que sea el Resumen de Cuenta Corriente Comercial de Credicoop.")
    st.stop()

saldo_ini = meta.get("saldo_inicial", np.nan)
saldo_fin = meta.get("saldo_final", np.nan)

total_deb = float(df["debito"].sum())
total_cred = float(df["credito"].sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Saldo anterior", fmt_ar(saldo_ini))
with c2:
    st.metric("Total créditos (+)", fmt_ar(total_cred))
with c3:
    st.metric("Total débitos (–)", fmt_ar(total_deb))

saldo_calc = np.nan
if not np.isnan(saldo_ini):
    saldo_calc = float(saldo_ini) + total_cred - total_deb

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("Saldo al (PDF)", fmt_ar(saldo_fin))
with c5:
    st.metric("Saldo final calculado", fmt_ar(saldo_calc))
diff = (saldo_calc - float(saldo_fin)) if (not np.isnan(saldo_calc) and not np.isnan(saldo_fin)) else np.nan
with c6:
    st.metric("Diferencia", fmt_ar(diff))

if not np.isnan(diff):
    st.success("Conciliado.") if abs(diff) < 0.01 else st.error("No cuadra la conciliación.")

# Resumen Operativo
st.caption("Resumen Operativo: Registración Módulo IVA")
df_res = build_resumen_operativo(df)
df_res_view = df_res.copy()
df_res_view["Importe"] = df_res_view["Importe"].map(fmt_ar)
st.dataframe(df_res_view, use_container_width=True, hide_index=True)

pdf_bytes = resumen_operativo_pdf_bytes(df_res)
if pdf_bytes:
    st.download_button(
        "📄 Descargar PDF – Resumen Operativo (Credicoop)",
        data=pdf_bytes,
        file_name="Resumen_Operativo_Credicoop.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# Préstamos
st.caption("Detalle de préstamos (acreditaciones / cuotas)")
df_prest = build_prestamos(df)
if df_prest.empty:
    st.info("No se detectaron préstamos en el período.")
else:
    df_pv = df_prest.copy()
    df_pv["debito"] = df_pv["debito"].map(fmt_ar)
    df_pv["credito"] = df_pv["credito"].map(fmt_ar)
    st.dataframe(df_pv, use_container_width=True, hide_index=True)

# Movimientos
st.caption("Detalle de movimientos")
df_view = df.copy()
df_view["debito"] = df_view["debito"].map(fmt_ar)
df_view["credito"] = df_view["credito"].map(fmt_ar)
st.dataframe(
    df_view[["fecha_raw", "comprobante", "descripcion", "Clasificación", "debito", "credito", "archivo", "pagina"]],
    use_container_width=True,
    hide_index=True,
)

# Descargas
st.caption("Descargar")
sheets = {
    "Movimientos": df[["fecha_raw", "fecha", "comprobante", "descripcion", "Clasificación", "debito", "credito", "archivo", "pagina"]],
    "Resumen_Operativo": df_res,
}
if not df_prest.empty:
    sheets["Prestamos"] = df_prest

try:
    xlsx_bytes = df_to_excel_bytes(sheets)
    st.download_button(
        "📥 Descargar Excel – Credicoop",
        data=xlsx_bytes,
        file_name="credicoop_movimientos_y_resumen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except Exception:
    st.download_button(
        "📥 Descargar CSV – Movimientos (fallback)",
        data=df_to_csv_bytes(df),
        file_name="credicoop_movimientos.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Herramienta para uso interno AIE San Justo | Developer Alfonso Alderete")
