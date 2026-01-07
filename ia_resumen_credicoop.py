# ia_resumen_credicoop.py
# IA Resumen Credicoop (PDF)
# Herramienta para uso interno - AIE San Justo
#
# - Conciliación estricta: Saldo anterior + Créditos − Débitos = Saldo al dd/mm/aaaa
# - Parser híbrido:
#   * PDFs con texto: parseo por coordenadas (page.chars)
#   * PDFs CID/Type3: OCR estructurado (pypdfium2 + tesseract)
# - Sin filtros de grilla: tablas HTML estáticas
# - Resumen Operativo + PDF (ReportLab) + Detalle de Préstamos (estricto)
#
# Developer: Alfonso Alderete

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- deps ----------------
try:
    import pdfplumber
except Exception as e:
    st.error(f"No se pudo importar pdfplumber: {e}\nRevisá requirements.txt")
    st.stop()

# OCR deps (solo se usan cuando hace falta)
try:
    import pytesseract
    from PIL import Image
    import pypdfium2 as pdfium
    OCR_OK = True
except Exception:
    OCR_OK = False

# PDF Resumen Operativo
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

def _inject_css():
    st.set_page_config(
        page_title="IA Resumen Credicoop",
        page_icon=str(FAVICON) if FAVICON.exists() else None,
        layout="centered",
    )
    st.markdown(
        """
        <style>
          .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 2rem; }
          /* tablas HTML */
          .aie-table-wrap { overflow-x: auto; border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; }
          table.aie-table { border-collapse: collapse; width: 100%; font-size: 13px; }
          table.aie-table th, table.aie-table td { padding: 8px 10px; border-bottom: 1px solid rgba(0,0,0,0.06); vertical-align: top; }
          table.aie-table th { background: rgba(0,0,0,0.04); text-align: left; font-weight: 600; }
          table.aie-table td.num { text-align: right; white-space: nowrap; }
          .aie-muted { color: rgba(0,0,0,0.65); }
        </style>
        """,
        unsafe_allow_html=True,
    )

def fmt_ar(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    try:
        return f"{float(n):,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")
    except Exception:
        return "—"

def _render_table(df: pd.DataFrame, money_cols: Optional[List[str]] = None):
    if df is None or df.empty:
        st.info("Sin datos para mostrar.")
        return
    df_show = df.copy()
    money_cols = money_cols or []
    for c in money_cols:
        if c in df_show.columns:
            df_show[c] = df_show[c].map(fmt_ar)

    html = df_show.to_html(index=False, escape=True, classes="aie-table")
    st.markdown(f'<div class="aie-table-wrap">{html}</div>', unsafe_allow_html=True)


# ---------------- parsing: money / dates ----------------
# soporta -1.234,56  |  1.234,56-  |  (1.234,56)
MONEY_RE = re.compile(r"(?<!\S)\(?-?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2}\)?-?(?!\S)")
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{2,4}\b")

def normalize_money(tok: str) -> float:
    if not tok:
        return np.nan
    s = tok.strip().replace("−", "-")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()
    if s.endswith("-"):
        neg = True
        s = s[:-1].strip()
    s = s.replace(".", "").replace(" ", "")
    if "," not in s:
        return np.nan
    main, frac = s.rsplit(",", 1)
    try:
        val = float(f"{main}.{frac}")
        return -val if neg else val
    except Exception:
        return np.nan

def norm_desc(desc: str) -> str:
    if not desc:
        return ""
    u = desc.upper()
    u = re.sub(r"\b\d{6,}\b", "", u)  # saca comprobantes largos
    u = " ".join(u.split())
    return u

def is_cid_pdf(pdf: pdfplumber.PDF) -> bool:
    try:
        t = pdf.pages[0].extract_text() or ""
        return "(cid:" in t
    except Exception:
        return False


# ---------------- classification ----------------
RE_PERCEP_IVA = re.compile(r"PERCEP|PERCEPCI", re.IGNORECASE)

def iva_bucket(desc_norm: str) -> str:
    if re.search(r"\b10[,\.]5\b", desc_norm) or "10,5" in desc_norm or "10.5" in desc_norm:
        return "10.5"
    return "21"

def clasificar(desc: str) -> str:
    n = norm_desc(desc)

    if "PREST" in n:
        return "Préstamos"

    if "25.413" in n or "25413" in n or "LEY 25.413" in n or "IMPUESTO LEY 25.413" in n:
        return "LEY 25.413"

    if RE_PERCEP_IVA.search(n) and "IVA" in n:
        return "Percepciones de IVA"

    if re.search(r"\bI\.?V\.?A\b", n) or " IVA " in f" {n} " or n.startswith("IVA"):
        if "PERCEP" in n:
            return "Percepciones de IVA"
        return "IVA"

    if "COMIS" in n or "GASTO" in n or "MANTEN" in n or "SERVICIO" in n or "CARGO" in n:
        return "Comisiones"

    if "EXENT" in n or "SELLAD" in n or "SEGURO" in n:
        return "Gastos Exentos"

    return "Otros"


# ---------------- Text-mode (chars) extraction ----------------
@dataclass
class Token:
    text: str
    x0: float
    x1: float
    top: float

def _tokens_from_chars(page) -> List[Token]:
    chars = page.chars or []
    toks: List[Token] = []
    for ch in chars:
        t = ch.get("text", "")
        if not t:
            continue
        if t == "\u00a0":
            t = " "
        toks.append(Token(t, float(ch["x0"]), float(ch["x1"]), float(ch["top"])))
    return toks

def _group_chars_into_lines(tokens: List[Token], y_tol: float = 2.0) -> List[List[Token]]:
    if not tokens:
        return []
    tokens = sorted(tokens, key=lambda t: (t.top, t.x0))
    lines: List[List[Token]] = []
    cur: List[Token] = []
    cur_y: Optional[float] = None
    for tk in tokens:
        if cur_y is None or abs(tk.top - cur_y) <= y_tol:
            cur.append(tk)
            if cur_y is None:
                cur_y = tk.top
        else:
            lines.append(sorted(cur, key=lambda t: t.x0))
            cur = [tk]
            cur_y = tk.top
    if cur:
        lines.append(sorted(cur, key=lambda t: t.x0))
    return lines

def _line_to_text(line: List[Token]) -> str:
    out = []
    prev_x1 = None
    for tk in line:
        if prev_x1 is None:
            out.append(tk.text)
        else:
            gap = tk.x0 - prev_x1
            out.append(" " if gap > 1.8 else "")
            out.append(tk.text)
        prev_x1 = tk.x1
    return "".join(out).strip()

def _extract_money_tokens_from_line(line: List[Token]) -> List[Tuple[float, str]]:
    chunks: List[Tuple[float, float, str]] = []
    cur = []
    cur_x0 = None
    cur_x1 = None
    prev_x1 = None
    for tk in line:
        if prev_x1 is not None and (tk.x0 - prev_x1) > 6.0:
            s = "".join(t.text for t in cur).strip()
            if s:
                chunks.append((cur_x0, cur_x1, s))
            cur = []
            cur_x0 = None
            cur_x1 = None
        cur.append(tk)
        cur_x0 = tk.x0 if cur_x0 is None else min(cur_x0, tk.x0)
        cur_x1 = tk.x1 if cur_x1 is None else max(cur_x1, tk.x1)
        prev_x1 = tk.x1
    s = "".join(t.text for t in cur).strip()
    if s:
        chunks.append((cur_x0 or 0.0, cur_x1 or 0.0, s))

    out = []
    for x0, x1, s in chunks:
        for m in MONEY_RE.finditer(s):
            out.append(((x0 + x1) / 2.0, m.group(0)))
    return out

def _kmeans_1d(points: List[float], k: int = 3, iters: int = 20) -> List[float]:
    pts = sorted(points)
    if len(pts) < k:
        return pts
    centers = [pts[int((i + 0.5) * len(pts) / k)] for i in range(k)]
    for _ in range(iters):
        groups = {i: [] for i in range(k)}
        for p in pts:
            j = min(range(k), key=lambda i: abs(p - centers[i]))
            groups[j].append(p)
        new_centers = []
        for i in range(k):
            new_centers.append(sum(groups[i]) / len(groups[i]) if groups[i] else centers[i])
        if all(abs(new_centers[i] - centers[i]) < 0.5 for i in range(k)):
            centers = new_centers
            break
        centers = new_centers
    return sorted(centers)

def _assign_amount_to_column(x_center: float, centers: List[float]) -> int:
    if not centers:
        return 0
    return int(min(range(len(centers)), key=lambda i: abs(x_center - centers[i])))

def _parse_text_pdf(pdf_bytes: bytes) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[str]]:
    rows = []
    saldo_anterior = None
    saldo_final = None
    saldo_final_label = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            txt = (page.extract_text() or "").upper()
            if "DETALLE" in txt and "TRANSFER" in txt:
                continue

            toks = _tokens_from_chars(page)
            lines = _group_chars_into_lines(toks, y_tol=2.3)

            amt_centers = []
            line_cache = []
            for line in lines:
                ltxt = _line_to_text(line)
                line_cache.append((line, ltxt))
                for xc, tok in _extract_money_tokens_from_line(line):
                    val = normalize_money(tok)
                    if not np.isnan(val):
                        amt_centers.append(xc)

            centers = _kmeans_1d(amt_centers, k=3) if len(amt_centers) >= 12 else sorted(amt_centers)[:3]

            for line, ltxt in line_cache:
                u = ltxt.upper()

                if saldo_anterior is None and "SALDO ANTERIOR" in u:
                    mm = MONEY_RE.search(ltxt)
                    if mm:
                        val = normalize_money(mm.group(0))
                        if not np.isnan(val):
                            saldo_anterior = float(val)

                if ("SALDO AL" in u or "SALDO FINAL" in u) and saldo_final is None:
                    mdate = DATE_RE.search(ltxt)
                    mm = MONEY_RE.search(ltxt)
                    if mm:
                        val = normalize_money(mm.group(0))
                        if not np.isnan(val):
                            saldo_final = float(val)
                            saldo_final_label = mdate.group(0) if mdate else None

                d = DATE_RE.search(ltxt)
                if not d or d.start() > 10:
                    continue

                amts = _extract_money_tokens_from_line(line)
                if not amts:
                    continue

                deb = 0.0
                cre = 0.0
                sal = np.nan

                for xc, tok in amts:
                    val = normalize_money(tok)
                    if np.isnan(val):
                        continue
                    col = _assign_amount_to_column(xc, centers)
                    if len(centers) >= 3:
                        if col == 0:
                            deb += abs(float(val))
                        elif col == 1:
                            cre += abs(float(val))
                        else:
                            sal = float(val)
                    elif len(centers) == 2:
                        if col == 0:
                            deb += abs(float(val))
                        else:
                            sal = float(val)
                    else:
                        deb += abs(float(val))

                m_first = MONEY_RE.search(ltxt)
                desc = ltxt[d.end(): m_first.start()].strip() if m_first else ltxt[d.end():].strip()
                desc = re.sub(r"\s{2,}", " ", desc).strip()

                if deb > 0 and cre > 0:
                    if deb >= cre:
                        cre = 0.0
                    else:
                        deb = 0.0

                rows.append({
                    "fecha": pd.to_datetime(d.group(0), dayfirst=True, errors="coerce").date(),
                    "fecha_raw": d.group(0),
                    "descripcion": desc,
                    "desc_norm": norm_desc(desc),
                    "debito": float(deb) if deb else 0.0,
                    "credito": float(cre) if cre else 0.0,
                    "saldo": sal,
                    "pagina": pi,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Clasificación"] = df["descripcion"].map(clasificar)
    return df, saldo_anterior, saldo_final, saldo_final_label


# ---------------- OCR mode (CID/Type3) ----------------
@dataclass
class OcrToken:
    text: str
    x0: float
    x1: float
    top: float

def _pdf_to_images(pdf_bytes: bytes, scale: float = 2.0) -> List[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil_img = page.render(scale=scale).to_pil()
        images.append(pil_img.convert("RGB"))
    return images

def _ocr_tokens(img: Image.Image, lang: str, config: str) -> List[OcrToken]:
    df = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).map(lambda s: s.strip())
    df = df[df["text"] != ""]
    out: List[OcrToken] = []
    for r in df.itertuples(index=False):
        out.append(OcrToken(str(r.text), float(r.left), float(r.left + r.width), float(r.top)))
    return out

def _group_ocr_lines(tokens: List[OcrToken], y_tol: float = 9.0) -> List[List[OcrToken]]:
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t.top, t.x0))
    lines: List[List[OcrToken]] = []
    cur: List[OcrToken] = []
    cur_y = None
    for t in toks:
        if cur_y is None or abs(t.top - cur_y) <= y_tol:
            cur.append(t)
            if cur_y is None:
                cur_y = t.top
        else:
            lines.append(sorted(cur, key=lambda x: x.x0))
            cur = [t]
            cur_y = t.top
    if cur:
        lines.append(sorted(cur, key=lambda x: x.x0))
    return lines

def _ocr_parse(pdf_bytes: bytes) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[str]]:
    if not OCR_OK:
        return pd.DataFrame(), None, None, None

    rows = []
    saldo_anterior = None
    saldo_final = None
    saldo_final_label = None

    images = _pdf_to_images(pdf_bytes, scale=2.0)

    global_centers = None

    for pi, img in enumerate(images, start=1):
        w, h = img.size
        header = img.crop((0, 0, w, int(h * 0.18)))
        head_txt = " ".join(t.text for t in _ocr_tokens(header, lang="spa", config="--psm 6"))[:500].upper()
        if "DETALLE" in head_txt and "TRANSFER" in head_txt:
            continue

        crop = img.crop((0, int(h * 0.16), w, int(h * 0.88)))

        toks_num = _ocr_tokens(crop, lang="eng", config="--psm 6 -c tessedit_char_whitelist=0123456789.,/()-")
        has_date = any(DATE_RE.search(t.text.replace("⁄", "/")) for t in toks_num)

        amt_centers = []
        for t in toks_num:
            m = MONEY_RE.search(t.text.replace(" ", ""))
            if not m:
                continue
            val = normalize_money(m.group(0))
            if not np.isnan(val):
                amt_centers.append((t.x0 + t.x1) / 2.0)

        if global_centers is None and len(amt_centers) >= 12:
            global_centers = _kmeans_1d([x for x in amt_centers], k=3)

        centers = global_centers or (_kmeans_1d([x for x in amt_centers], k=3) if len(amt_centers) >= 12 else sorted([x for x in amt_centers])[:3])

        desc_crop = img.crop((0, int(h * 0.16), int(w * 0.72), int(h * 0.95)))
        toks_txt = _ocr_tokens(desc_crop, lang="spa", config="--psm 6")

        merged: List[OcrToken] = []
        crop_y0 = int(h * 0.16)
        desc_y0 = int(h * 0.16)

        for t in toks_num:
            merged.append(OcrToken(t.text, t.x0, t.x1, t.top + 0.0))
        for t in toks_txt:
            merged.append(OcrToken(t.text, t.x0, t.x1, t.top + (desc_y0 - crop_y0)))

        lines = _group_ocr_lines(merged, y_tol=9.5)

        for line in lines:
            ltxt = " ".join(t.text for t in line)
            u = ltxt.upper()

            if saldo_anterior is None and "SALDO" in u and "ANTER" in u:
                mm = MONEY_RE.search(ltxt)
                if mm:
                    val = normalize_money(mm.group(0))
                    if not np.isnan(val):
                        saldo_anterior = float(val)

            if saldo_final is None and "SALDO" in u and (" AL " in f" {u} " or "FINAL" in u):
                mm = MONEY_RE.search(ltxt)
                if mm:
                    val = normalize_money(mm.group(0))
                    if not np.isnan(val):
                        saldo_final = float(val)
                        mdate = DATE_RE.search(ltxt)
                        saldo_final_label = mdate.group(0) if mdate else None

            if not has_date:
                continue

            d = DATE_RE.search(ltxt.replace("⁄", "/"))
            if not d or d.start() > 12:
                continue

            amts = []
            for t in line:
                m = MONEY_RE.search(t.text.replace(" ", ""))
                if not m:
                    continue
                if DATE_RE.fullmatch(t.text.replace("⁄", "/")):
                    continue
                val = normalize_money(m.group(0))
                if np.isnan(val):
                    continue
                amts.append(((t.x0 + t.x1) / 2.0, abs(float(val))))

            if not amts:
                continue

            deb = 0.0
            cre = 0.0

            for xc, val in amts:
                col = _assign_amount_to_column(xc, centers)
                if len(centers) >= 3:
                    if col == 0:
                        deb += val
                    elif col == 1:
                        cre += val
                elif len(centers) == 2:
                    if col == 0:
                        deb += val

            desc_right = (centers[0] + centers[1]) / 2.0 if len(centers) >= 2 else (w * 0.7)
            desc_parts = []
            for t in line:
                tt = t.text.strip()
                if not tt:
                    continue
                if DATE_RE.fullmatch(tt.replace("⁄", "/")):
                    continue
                if MONEY_RE.fullmatch(tt.replace(" ", "")):
                    continue
                if (t.x1) <= desc_right:
                    desc_parts.append(tt)
            desc = " ".join(desc_parts)
            desc = re.sub(r"\s{2,}", " ", desc).strip()

            if deb > 0 and cre > 0:
                if deb >= cre:
                    cre = 0.0
                else:
                    deb = 0.0

            rows.append({
                "fecha": pd.to_datetime(d.group(0), dayfirst=True, errors="coerce").date(),
                "fecha_raw": d.group(0),
                "descripcion": desc,
                "desc_norm": norm_desc(desc),
                "debito": float(deb) if deb else 0.0,
                "credito": float(cre) if cre else 0.0,
                "saldo": np.nan,
                "pagina": pi,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Clasificación"] = df["descripcion"].map(clasificar)
    return df, saldo_anterior, saldo_final, saldo_final_label


# ---------------- Resumen Operativo ----------------
def build_resumen_operativo(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Concepto", "Débitos", "Créditos", "Total (Débitos - Créditos)"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df["Clasificación"] = df["Clasificación"].fillna("Otros")
    df["desc_norm"] = df["desc_norm"].fillna("")

    m_ley = df["Clasificación"].eq("LEY 25.413")
    m_piva = df["Clasificación"].eq("Percepciones de IVA")
    m_ex = df["Clasificación"].eq("Gastos Exentos")
    m_iva = df["Clasificación"].eq("IVA")
    m_com = df["Clasificación"].eq("Comisiones")

    buck = df["desc_norm"].map(iva_bucket)
    iva_105 = m_iva & buck.eq("10.5")
    iva_21 = m_iva & ~buck.eq("10.5")

    com_105 = m_com & buck.eq("10.5")
    com_21 = m_com & ~buck.eq("10.5")

    def sums(mask):
        d = float(df.loc[mask, "debito"].sum())
        c = float(df.loc[mask, "credito"].sum())
        return d, c, d - c

    rows = []
    d, c, t = sums(com_21); rows.append(["Comisiones (Gastos Bancarios) Neto", d, c, t])
    d, c, t = sums(iva_21); rows.append(["IVA", d, c, t])
    d, c, t = sums(com_105); rows.append(["Gastos Bancarios 10.5", d, c, t])
    d, c, t = sums(iva_105); rows.append(["IVA 10.5", d, c, t])
    d, c, t = sums(m_ley); rows.append(["Impuesto Ley 25.413", d, c, t])
    d, c, t = sums(m_piva); rows.append(["Percepciones de IVA", d, c, t])
    d, c, t = sums(m_ex); rows.append(["Gastos Exentos", d, c, t])

    d = float(df["debito"].sum())
    c = float(df["credito"].sum())
    rows.append(["Total", d, c, d - c])

    return pd.DataFrame(rows, columns=cols)

def build_prestamos(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    m = df["Clasificación"].eq("Préstamos")
    cols = ["fecha_raw", "descripcion", "debito", "credito", "pagina"]
    return df.loc[m, cols].copy()


# ---------------- downloads ----------------
def df_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for name, d in sheets.items():
            d.to_excel(writer, index=False, sheet_name=name[:31] or "Sheet1")

        wb = writer.book
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        for name, d in sheets.items():
            ws = writer.sheets[name[:31] or "Sheet1"]
            for idx, col in enumerate(d.columns):
                width = min(max(len(str(col)), 12) + 2, 45)
                ws.set_column(idx, idx, width)
            for colname in ["debito", "credito", "saldo", "Débitos", "Créditos", "Total (Débitos - Créditos)"]:
                if colname in d.columns:
                    j = list(d.columns).index(colname)
                    ws.set_column(j, j, 18, money_fmt)
    return out.getvalue()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def build_resumen_operativo_pdf(df_op: pd.DataFrame, title: str = "Resumen Operativo – Credicoop") -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None
    try:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title=title)
        styles = getSampleStyleSheet()
        elems = [Paragraph(title, styles["Title"]), Spacer(1, 10)]

        data = [list(df_op.columns)]
        for _, r in df_op.iterrows():
            data.append([
                str(r["Concepto"]),
                fmt_ar(r["Débitos"]),
                fmt_ar(r["Créditos"]),
                fmt_ar(r["Total (Débitos - Créditos)"]),
            ])

        tbl = Table(data, colWidths=[260, 95, 95, 110])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ]))
        elems.append(tbl)
        elems.append(Spacer(1, 12))
        elems.append(Paragraph("Herramienta para uso interno - AIE San Justo", styles["Normal"]))
        doc.build(elems)
        return pdf_buf.getvalue()
    except Exception:
        return None


# ---------------- app main ----------------
def parse_any(pdf_bytes: bytes) -> Tuple[pd.DataFrame, float, float, Optional[str], str]:
    """
    Returns: df, saldo_anterior, saldo_final, saldo_final_label, mode
    mode: 'text' | 'ocr' | 'none'
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        cid = is_cid_pdf(pdf)

    if not cid:
        df, sa, sf, sfl = _parse_text_pdf(pdf_bytes)
        if df is not None and not df.empty and sa is not None and sf is not None:
            return df, float(sa), float(sf), sfl, "text"

    df, sa, sf, sfl = _ocr_parse(pdf_bytes)
    if df is not None and not df.empty and sa is not None and sf is not None:
        return df, float(sa), float(sf), sfl, "ocr"

    # diagnostics fallback
    if not cid:
        df, sa, sf, sfl = _parse_text_pdf(pdf_bytes)
        return df, float(sa or 0.0), float(sf or 0.0), sfl, "text"

    return pd.DataFrame(), 0.0, 0.0, None, "none"


def run_app():
    _inject_css()

    if LOGO.exists():
        st.image(str(LOGO), width=200)

    st.title("IA Resumen Credicoop")
    st.caption("Conciliación estricta + Resumen Operativo + Préstamos. Sin filtros de grilla.")

    up = st.file_uploader("Subí UN PDF de Credicoop", type=["pdf"], accept_multiple_files=False)
    if up is None:
        st.info("La app no almacena datos. Procesa el PDF en memoria.")
        return

    pdf_bytes = up.read()

    with st.spinner("Procesando PDF..."):
        df, saldo_anterior, saldo_final, saldo_label, mode = parse_any(pdf_bytes)

    if df.empty:
        st.error(
            "No se pudieron extraer movimientos de este PDF.\n\n"
            "Si es un PDF CID/Type3, verificá que Tesseract esté instalado (packages.txt)."
        )
        st.stop()

    df = df.dropna(subset=["fecha_raw"]).copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha_raw"], dayfirst=True, errors="coerce")
    df = df.sort_values(["fecha_dt", "pagina"]).reset_index(drop=True)

    total_deb = float(df["debito"].sum())
    total_cre = float(df["credito"].sum())
    calc_final = float(saldo_anterior) + total_cre - total_deb
    diff = calc_final - float(saldo_final)
    cuadra = abs(diff) < 0.01

    st.caption("Resumen del período")
    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo anterior", f"$ {fmt_ar(saldo_anterior)}")
    c2.metric("Total créditos (+)", f"$ {fmt_ar(total_cre)}")
    c3.metric("Total débitos (–)", f"$ {fmt_ar(total_deb)}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Saldo al (PDF)", f"$ {fmt_ar(saldo_final)}")
    c5.metric("Saldo calculado", f"$ {fmt_ar(calc_final)}")
    c6.metric("Diferencia", f"$ {fmt_ar(diff)}")

    if saldo_label:
        st.caption(f"Saldo final tomado de: {saldo_label}")
    st.caption(f"Modo de lectura: {mode.upper()}")

    st.success("Conciliado.") if cuadra else st.error("No cuadra la conciliación (estricto).")

    st.markdown("---")
    st.subheader("Resumen Operativo")
    df_op = build_resumen_operativo(df)
    _render_table(df_op, money_cols=["Débitos", "Créditos", "Total (Débitos - Créditos)"])

    pdf_op = build_resumen_operativo_pdf(df_op, title="Resumen Operativo: Registración Módulo IVA (Credicoop)")
    if pdf_op:
        st.download_button(
            "📄 Descargar PDF – Resumen Operativo",
            data=pdf_op,
            file_name="Resumen_Operativo_Credicoop.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("No se pudo generar PDF del Resumen Operativo (verificar ReportLab).")

    st.markdown("---")
    st.subheader("Préstamos (detección estricta)")
    df_prest = build_prestamos(df)
    if df_prest.empty:
        st.info("No se detectaron préstamos (solo descripciones con 'PREST').")
    else:
        _render_table(df_prest, money_cols=["debito", "credito"])

    st.markdown("---")
    st.subheader("Movimientos")
    df_view = df[["fecha_raw", "descripcion", "Clasificación", "debito", "credito", "pagina"]].copy()
    _render_table(df_view, money_cols=["debito", "credito"])

    st.markdown("---")
    st.subheader("Descargas")
    xlsx = df_to_excel_bytes({
        "Movimientos": df.drop(columns=["fecha_dt"], errors="ignore"),
        "Resumen_Operativo": df_op,
        "Prestamos": df_prest if not df_prest.empty else pd.DataFrame(columns=df_prest.columns),
    })
    st.download_button(
        "📥 Descargar Excel (XLSX)",
        data=xlsx,
        file_name="credicoop_movimientos_y_resumen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.download_button(
        "📥 Descargar Movimientos (CSV)",
        data=df_to_csv_bytes(df.drop(columns=["fecha_dt"], errors="ignore")),
        file_name="credicoop_movimientos.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "📥 Descargar Resumen Operativo (CSV)",
        data=df_to_csv_bytes(df_op),
        file_name="credicoop_resumen_operativo.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("Herramienta para uso interno AIE San Justo | Developer Alfonso Alderete")
