# streamlit_app.py
# Herramienta para uso interno - AIE San Justo
# Developer: Alfonso Alderete
# VERSIÓN: Spatial Strict - Corregida para Streamlit Cloud

import io
import re
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------- CONFIGURACIÓN E IMPORTACIONES ----------------
st.set_page_config(
    page_title="IA Resumen Credicoop",
    layout="centered",
)

# Estilos CSS (Estética Original)
st.markdown("""
    <style>
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Importación de librerías
try:
    import pdfplumber
    import xlsxwriter
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
except ImportError as e:
    st.error(f"⚠️ Error: Falta la librería {e}. Asegurate de que requirements.txt tenga: pdfplumber, pandas, xlsxwriter, reportlab, streamlit")
    st.stop()

# ---------------- LOGO ----------------
HERE = Path(__file__).parent
LOGO = HERE / "logo_aie.png"
if LOGO.exists():
    st.image(str(LOGO), width=200)

st.title("IA Resumen Credicoop")

# ---------------- FUNCIONES DE PARSEO ----------------

def parse_currency(text):
    """Convierte texto (1.234,56) a float. Maneja negativos."""
    if not text: return 0.0
    # Limpia todo lo que no sea dígito, coma, punto o menos
    clean = re.sub(r'[^\d,.-]', '', str(text))
    # Detecta negativo al final (común en bancos: 100.00-)
    is_negative = "-" in clean
    clean = clean.replace("-", "")
    
    try:
        # Formato Argentino: 1.234,56 -> 1234.56
        if "," in clean:
            clean = clean.replace(".", "").replace(",", ".")
        val = float(clean)
        return -val if is_negative else val
    except ValueError:
        return 0.0

def fmt_ar(n):
    """Formato visual Argentino."""
    if pd.isna(n) or n is None: return "—"
    return "{:,.2f}".format(n).replace(",", "X").replace(".", ",").replace("X", ".")

def clasificar(desc):
    """Clasifica el movimiento para el Resumen Operativo."""
    d = str(desc).upper()
    if "25413" in d or "25.413" in d or "LEY 25413" in d: return "Ley 25.413"
    if "SIRCREB" in d: return "SIRCREB"
    if "IVA" in d and ("PERCEP" in d or "RG" in d or "2408" in d): return "Percepciones de IVA"
    if "IVA" in d and "DEBITO FISCAL" in d:
        if "10,5" in d or "10.5" in d: return "IVA 10,5%"
        return "IVA 21%"
    if ("INTERES" in d or "SALDO DEUDOR" in d) and "IVA" not in d: return "Comisiones/Gastos Neto 10,5%"
    if any(k in d for k in ["COMISION", "SERVICIO", "MANTEN", "GASTO", "CARGO"]) and "IVA" not in d:
        return "Comisiones/Gastos Neto 21%"
    if "PREST" in d or "MUTUO" in d: return "Préstamos"
    return "Otros"

# ---------------- LÓGICA DE EXTRACCIÓN (CORE) ----------------

def extract_data_pdf(pdf_bytes, filename):
    movements = []
    resumen_items = []
    saldo_inicial = np.nan
    saldo_final_pdf = np.nan
    fecha_cierre = ""
    
    # Punto de corte (eje X) por defecto. Se ajusta si hay encabezados.
    split_point = 420 
    
    capturando_resumen = False

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # Extraer palabras con sus coordenadas
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=True)
            
            # 1. Calibración dinámica de columnas
            header_deb = next((w for w in words if "DEBITO" in w['text'].upper()), None)
            header_cred = next((w for w in words if "CREDITO" in w['text'].upper()), None)
            
            if header_deb and header_cred:
                # El corte es el promedio entre el fin de DEBITO y el inicio de CREDITO
                split_point = (header_deb['x1'] + header_cred['x0']) / 2

            # 2. Agrupar palabras por líneas visuales
            lines = {}
            for w in words:
                y = round(w['top'])
                if y not in lines: lines[y] = []
                lines[y].append(w)
            
            sorted_y = sorted(lines.keys())
            last_idx = -1

            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words]).strip()
                line_text_upper = line_text.upper()

                # Ignorar encabezados repetidos
                if "FECHA" in line_text_upper and "SALDO" in line_text_upper:
                    continue

                # --- A. SALDOS GLOBALES ---
                if "SALDO ANTERIOR" in line_text_upper:
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_inicial = parse_currency(nums[-1]['text'])
                    continue
                
                if "SALDO AL" in line_text_upper:
                    # Capturar fecha
                    parts = line_text_upper.split("SALDO AL")
                    if len(parts) > 1:
                        fecha_cierre = parts[1].strip().split(" ")[0]
                    # Capturar saldo final
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_final_pdf = parse_currency(nums[-1]['text'])
                    capturando_resumen = True
                    continue

                # --- B. RESUMEN OPERATIVO (TABLA FINAL) ---
                if capturando_resumen:
                    if "LIQUIDACION" in line_text_upper or "DEBITO DIRECTO" in line_text_upper:
                        capturando_resumen = False
                    else:
                        nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                        if nums:
                            monto_obj = nums[-1]
                            # El concepto es todo el texto MENOS el número
                            concepto = " ".join([w['text'] for w in line_words if w != monto_obj])
                            if len(concepto) > 3 and "CREDITO DE IMPUESTO" not in concepto.upper():
                                resumen_items.append({"Concepto": concepto, "Importe": parse_currency(monto_obj['text'])})
                    continue

                # --- C. MOVIMIENTOS ---
                first_word = line_words[0]['text']
                # Regex fecha (dd/mm/yy o dd/mm/yyyy)
                is_date = re.match(r'\d{2}/\d{2}/\d{2}', first_word)

                if is_date:
                    fecha_raw = first_word
                    try:
                        dt = datetime.strptime(fecha_raw[:8], "%d/%m/%y").date()
                    except:
                        try:
                            dt = datetime.strptime(fecha_raw[:10], "%d/%m/%Y").date()
                        except:
                            dt = None

                    money_tokens = []
                    text_tokens = []
                    comprobante = ""
                    
                    for w in line_words[1:]: # Saltar la fecha
                        txt = w['text']
                        if re.match(r'^-?[\d\.]+,[\d]{2}$', txt):
                            money_tokens.append(w)
                        elif re.match(r'^\d{4,}$', txt) and "," not in txt:
                            comprobante = txt
                        else:
                            text_tokens.append(txt)
                    
                    descripcion = " ".join(text_tokens)
                    debito = 0.0
                    credito = 0.0
                    
                    # --- REGLA: MONTOS vs SALDOS ---
                    # Si hay montos, el de la derecha es saldo (se ignora).
                    # El de la izquierda es el movimiento real.
                    if money_tokens:
                        money_tokens.sort(key=lambda w: w['x0']) # Ordenar izq a der
                        
                        token_real = money_tokens[0] # El primero es el movimiento
                        valor = parse_currency(token_real['text'])
                        
                        # Decidir si es débito o crédito por posición X
                        center_x = (token_real['x0'] + token_real['x1']) / 2
                        
                        if center_x < split_point:
                            debito = valor
                        else:
                            credito = valor
                    
                    movements.append({
                        "fecha": dt,
                        "fecha_raw": fecha_raw,
                        "comprobante": comprobante,
                        "descripcion": descripcion,
                        "debito": debito,
                        "credito": credito,
                        "pagina": page.page_number
                    })
                    last_idx = len(movements) - 1

                elif last_idx >= 0 and not capturando_resumen:
                    # Continuación de descripción
                    movements[last_idx]["descripcion"] += " " + line_text

    meta = {
        "saldo_inicial": saldo_inicial,
        "saldo_final": saldo_final_pdf,
        "fecha_final": fecha_cierre
    }
    return pd.DataFrame(movements), meta, resumen_items

# ---------------- GENERACIÓN DE REPORTES ----------------

def df_to_excel_bytes(sheets):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_money = wb.add_format({"num_format": "#,##0.00"})
        
        for name, df in sheets.items():
            if df.empty: continue
            df.to_excel(writer, index=False, sheet_name=name[:30])
            ws = writer.sheets[name[:30]]
            for i, col in enumerate(df.columns):
                ws.set_column(i, i, 15)
            # Formato moneda
            for col_name in ["debito", "credito", "Importe"]:
                if col_name in df.columns:
                    idx = df.columns.get_loc(col_name)
                    ws.set_column(idx, idx, 18, fmt_money)
    return out.getvalue()

def resumen_pdf_bytes(df_res):
    if df_res.empty: return None
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title="Resumen Operativo")
    styles = getSampleStyleSheet()
    elems = []
    
    elems.append(Paragraph("Resumen Operativo: Registración Módulo IVA", styles["Title"]))
    elems.append(Spacer(1, 10))
    
    data = [["Concepto", "Importe"]] + [[r["Concepto"], fmt_ar(r["Importe"])] for r in df_res.to_dict('records')]
    total = df_res["Importe"].sum()
    data.append(["TOTAL", fmt_ar(total)])
    
    tbl = Table(data, colWidths=[340, 160])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 20))
    elems.append(Paragraph("Herramienta para uso interno AIE San Justo", styles["Normal"]))
    
    doc.build(elems)
    return pdf_buf.getvalue()

# ---------------- APP PRINCIPAL ----------------

uploaded = st.file_uploader("Subí un PDF del resumen bancario (Banco Credicoop)", type=["pdf"])

if uploaded is not None:
    with st.spinner("Procesando PDF..."):
        try:
            pdf_bytes = uploaded.read()
            df, meta, lista_resumen = extract_data_pdf(pdf_bytes, uploaded.name)
            
            if df.empty:
                st.error("El archivo se leyó pero no se encontraron movimientos. Verificá que sea un PDF original.")
            else:
                # 1. Clasificar
                df["Clasificación"] = df["descripcion"].apply(clasificar)
                
                total_deb = df["debito"].sum()
                total_cred = df["credito"].sum()
                saldo_ini = meta["saldo_inicial"]
                saldo_fin_pdf = meta["saldo_final"]
                
                saldo_calc = np.nan
                diff = np.nan
                if not np.isnan(saldo_ini):
                    saldo_calc = saldo_ini + total_cred - total_deb
                    if not np.isnan(saldo_fin_pdf):
                        diff = saldo_calc - saldo_fin_pdf

                # 2. Métricas
                c1, c2, c3 = st.columns(3)
                c1.metric("Saldo Anterior", fmt_ar(saldo_ini))
                c2.metric("Total Créditos", fmt_ar(total_cred))
                c3.metric("Total Débitos", fmt_ar(total_deb))

                c4, c5, c6 = st.columns(3)
                c4.metric("Saldo PDF", fmt_ar(saldo_fin_pdf))
                c5.metric("Saldo Calculado", fmt_ar(saldo_calc))
                
                if not np.isnan(diff):
                    st_concil = "Conciliado" if abs(diff) < 1.0 else "Diferencia"
                    color = "normal" if abs(diff) < 1.0 else "inverse"
                    c6.metric("Estado", st_concil, delta=fmt_ar(diff), delta_color=color)

                # 3. Resumen Operativo
                st.markdown("---")
                st.caption("Resumen Operativo: Registración Módulo IVA")
                if lista_resumen:
                    df_res = pd.DataFrame(lista_resumen)
                    st.success("Tabla detectada automáticamente.")
                else:
                    st.warning("Calculado desde movimientos.")
                    df_res = df[df["debito"] > 0].groupby("Clasificación")["debito"].sum().reset_index()
                    df_res.rename(columns={"debito": "Importe", "Clasificación": "Concepto"}, inplace=True)

                st.dataframe(df_res.style.format({"Importe": "{:,.2f}"}), use_container_width=True)
                
                pdf_res = resumen_pdf_bytes(df_res)
                if pdf_res:
                    st.download_button("📄 Descargar PDF Resumen", pdf_res, "Resumen_Operativo.pdf", "application/pdf", use_container_width=True)

                # 4. Préstamos
                df_prest = df[df["Clasificación"] == "Préstamos"].copy()
                if not df_prest.empty:
                    st.caption("Préstamos Bancarios")
                    st.dataframe(df_prest, use_container_width=True)

                # 5. Grilla Completa
                st.caption("Detalle de movimientos")
                st.dataframe(df[["fecha_raw", "comprobante", "descripcion", "Clasificación", "debito", "credito"]].style.format({"debito": "{:,.2f}", "credito": "{:,.2f}"}), use_container_width=True)

                # 6. Descarga Excel
                sheets = {"Movimientos": df, "Resumen": df_res, "Prestamos": df_prest}
                st.download_button("📥 Descargar Excel Completo", df_to_excel_bytes(sheets), "Credicoop_Completo.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        except Exception as e:
            st.error(f"Error procesando: {e}")
