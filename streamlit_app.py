# streamlit_app.py
# Herramienta para uso interno - AIE San Justo
# Developer: Alfonso Alderete
# VERSIÓN: Bounding Box Logic (Ignora encabezados estrictamente)

import io
import re
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(
    page_title="IA Resumen Credicoop",
    layout="centered",
)

st.markdown("""
    <style>
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 2rem; }
      h1 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# ---------------- LIBRERÍAS ----------------
try:
    import pdfplumber
    import xlsxwriter
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
except ImportError as e:
    st.error(f"Falta librería: {e}")
    st.stop()

# ---------------- ASSETS ----------------
HERE = Path(__file__).parent
LOGO = HERE / "logo_aie.png"
if LOGO.exists():
    st.image(str(LOGO), width=200)

st.title("IA Resumen Credicoop")

# ---------------- PARSEO Y LÓGICA ----------------

def parse_currency(text):
    """Convierte $ 1.000,00 a float 1000.00"""
    if not text: return 0.0
    clean = re.sub(r'[^\d,.-]', '', str(text))
    # Negativos al final (ej: 100.00-)
    is_negative = "-" in clean
    clean = clean.replace("-", "")
    try:
        if "," in clean:
            clean = clean.replace(".", "").replace(",", ".")
        val = float(clean)
        return -val if is_negative else val
    except ValueError:
        return 0.0

def fmt_ar(n):
    if pd.isna(n) or n is None: return "—"
    return "{:,.2f}".format(n).replace(",", "X").replace(".", ",").replace("X", ".")

def clasificar(desc):
    d = str(desc).upper()
    if "25413" in d or "25.413" in d: return "Impuesto Ley 25.413"
    if "SIRCREB" in d: return "SIRCREB"
    if "IVA" in d and ("PERCEP" in d or "RG" in d or "2408" in d): return "Percepciones de IVA"
    if "IVA" in d and "DEBITO FISCAL" in d:
        if "10,5" in d: return "IVA 10,5%"
        return "IVA 21%"
    if ("INTERES" in d or "SALDO DEUDOR" in d) and "IVA" not in d: return "Comisiones/Gastos Neto 10,5%"
    if any(k in d for k in ["COMISION", "SERVICIO", "MANTEN", "GASTO", "CARGO"]) and "IVA" not in d:
        return "Comisiones/Gastos Neto 21%"
    if "PREST" in d or "MUTUO" in d: return "Préstamos"
    return "Otros"

def extract_data_clean(pdf_bytes, filename):
    movements = []
    resumen_items = []
    
    saldo_inicial = np.nan
    saldo_final_pdf = np.nan
    
    # Coordenada X para dividir Debito/Credito (se auto-calibra)
    split_point = 420 
    
    capturando_resumen = False

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            width = page.width
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=True)
            
            # --- PASO 1: DEFINIR EL ÁREA ÚTIL (Sacar encabezado) ---
            # Buscamos la coordenada Y donde termina el encabezado de la tabla
            header_y_limit = 0
            
            # Buscamos palabras clave de cabecera
            header_deb = next((w for w in words if "DEBITO" in w['text'].upper()), None)
            header_cred = next((w for w in words if "CREDITO" in w['text'].upper()), None)
            header_fecha = next((w for w in words if "FECHA" in w['text'].upper()), None)
            
            if header_fecha:
                header_y_limit = header_fecha['bottom'] + 5
            elif page.page_number > 1:
                # Si es pagina 2+, y no encuentra header, asumimos margen superior estandar
                header_y_limit = 100 
            
            # Calibrar columnas si encontramos los titulos
            if header_deb and header_cred:
                split_point = (header_deb['x1'] + header_cred['x0']) / 2
                header_y_limit = max(header_y_limit, header_deb['bottom'])

            # --- PASO 2: FILTRAR PALABRAS ---
            # Solo nos interesan las palabras que están DEBAJO del encabezado
            # Esto elimina logo, dirección, CUIT del cliente, etc.
            content_words = [w for w in words if w['top'] > header_y_limit]
            
            # Agrupar por líneas
            lines = {}
            for w in content_words:
                y = round(w['top'])
                if y not in lines: lines[y] = []
                lines[y].append(w)
            
            sorted_y = sorted(lines.keys())
            last_idx = -1

            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words]).strip()
                line_text_upper = line_text.upper()

                # Ignorar pies de página comunes
                if "CONTINUA EN PAGINA" in line_text_upper or "VIENE DE PAGINA" in line_text_upper:
                    continue
                if "TOTALES" in line_text_upper and len(line_words) < 5:
                    continue

                # --- A. SALDOS (Detectados por texto específico) ---
                if "SALDO ANTERIOR" in line_text_upper:
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_inicial = parse_currency(nums[-1]['text'])
                    continue # No es un movimiento, saltar
                
                if "SALDO AL" in line_text_upper:
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_final_pdf = parse_currency(nums[-1]['text'])
                    capturando_resumen = True
                    continue # No es un movimiento, saltar

                # --- B. RESUMEN DE IMPUESTOS (Al final) ---
                if capturando_resumen:
                    if "LIQUIDACION" in line_text_upper or "DEBITO DIRECTO" in line_text_upper:
                        capturando_resumen = False
                    else:
                        nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                        if nums:
                            monto_obj = nums[-1]
                            concepto = " ".join([w['text'] for w in line_words if w != monto_obj])
                            # Filtro para no meter basura
                            if len(concepto) > 3 and "CREDITO DE IMPUESTO" not in concepto.upper():
                                resumen_items.append({"Concepto": concepto, "Importe": parse_currency(monto_obj['text'])})
                    continue

                # --- C. MOVIMIENTOS REALES ---
                # Un movimiento REAL empieza con una fecha a la izquierda.
                # Regex estricto: Inicio de linea o muy cerca.
                date_match = re.search(r'^\s*\d{2}[/.-]\d{2}[/.-](\d{2}|\d{4})', line_text)
                
                if date_match:
                    fecha_raw = date_match.group(0).strip()
                    try:
                        dt = datetime.strptime(fecha_raw[:8], "%d/%m/%y").date()
                    except:
                        try:
                            dt = datetime.strptime(fecha_raw[:10], "%d/%m/%Y").date()
                        except:
                            dt = None

                    # Identificar partes de la línea
                    money_tokens = []
                    text_parts = []
                    comprobante = ""
                    
                    for w in line_words:
                        txt = w['text']
                        # Si es la fecha, la saltamos
                        if txt in fecha_raw: continue
                        
                        # Es dinero? (Debe tener coma decimal)
                        if re.match(r'^-?[\d\.]+,[\d]{2}$', txt):
                            money_tokens.append(w)
                        # Es comprobante? (Numeros largos sin puntuacion)
                        elif re.match(r'^\d{4,}$', txt) and "," not in txt:
                            comprobante = txt
                        else:
                            text_parts.append(txt)
                    
                    descripcion = " ".join(text_parts)
                    debito = 0.0
                    credito = 0.0
                    
                    # --- REGLA DE ORO DEL USUARIO ---
                    # Si hay montos, tomamos el PRIMERO (izquierda). El resto es saldo.
                    if money_tokens:
                        money_tokens.sort(key=lambda w: w['x0'])
                        token_real = money_tokens[0]
                        valor = parse_currency(token_real['text'])
                        
                        # Decidir columna por posición X
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
                    # Línea de continuación (Descripción larga)
                    # SOLO si no parece un encabezado basura
                    if "FECHA" not in line_text_upper and "SALDO" not in line_text_upper:
                        movements[last_idx]["descripcion"] += " " + line_text

    meta = {"saldo_inicial": saldo_inicial, "saldo_final": saldo_final_pdf}
    return pd.DataFrame(movements), meta, resumen_items

# ---------------- GENERADORES ----------------

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
                ws.set_column(i, i, 18, fmt_money if col in ["debito", "credito", "Importe"] else None)
    return out.getvalue()

def resumen_pdf_bytes(df_res):
    if df_res.empty: return None
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4)
    elems = [Paragraph("Resumen Operativo IVA", getSampleStyleSheet()["Title"]), Spacer(1, 10)]
    data = [["Concepto", "Importe"]] + [[r["Concepto"], fmt_ar(r["Importe"])] for r in df_res.to_dict('records')]
    data.append(["TOTAL", fmt_ar(df_res["Importe"].sum())])
    t = Table(data, colWidths=[340, 160])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]))
    elems.append(t)
    doc.build(elems)
    return pdf_buf.getvalue()

# ---------------- APP MAIN ----------------

uploaded = st.file_uploader("Subí un PDF del resumen bancario (Banco Credicoop)", type=["pdf"])

if uploaded is not None:
    with st.spinner("Procesando PDF (Filtrando encabezados y validando saldos)..."):
        try:
            pdf_bytes = uploaded.read()
            df, meta, lista_resumen = extract_data_clean(pdf_bytes, uploaded.name)
            
            if df.empty:
                st.error("No se detectaron movimientos. Verificá que el archivo sea el correcto.")
            else:
                df["Clasificación"] = df["descripcion"].apply(clasificar)
                
                # Totales
                t_deb = df["debito"].sum()
                t_cred = df["credito"].sum()
                s_ini = meta["saldo_inicial"]
                s_fin = meta["saldo_final"]
                
                # Conciliación
                s_calc = np.nan
                diff = np.nan
                if not np.isnan(s_ini):
                    s_calc = s_ini + t_cred - t_deb
                    if not np.isnan(s_fin):
                        diff = s_calc - s_fin

                # Métricas
                c1, c2, c3 = st.columns(3)
                c1.metric("Saldo Anterior", fmt_ar(s_ini))
                c2.metric("Créditos (+)", fmt_ar(t_cred))
                c3.metric("Débitos (-)", fmt_ar(t_deb))
                
                c4, c5, c6 = st.columns(3)
                c4.metric("Saldo PDF", fmt_ar(s_fin))
                c5.metric("Saldo Calculado", fmt_ar(s_calc))
                if not np.isnan(diff):
                    c6.metric("Diferencia", fmt_ar(diff), delta="Conciliado" if abs(diff)<1 else "Error", delta_color="normal" if abs(diff)<1 else "inverse")

                # Resumen Operativo
                st.markdown("---")
                st.subheader("Resumen Operativo")
                if lista_resumen:
                    df_res = pd.DataFrame(lista_resumen)
                    st.info("Tabla de impuestos extraída del PDF.")
                else:
                    st.warning("Calculando impuestos desde movimientos.")
                    df_res = df[df["debito"] > 0].groupby("Clasificación")["debito"].sum().reset_index()
                    df_res.rename(columns={"debito": "Importe", "Clasificación": "Concepto"}, inplace=True)
                
                st.dataframe(df_res.style.format({"Importe": "{:,.2f}"}), use_container_width=True)
                pdf_res = resumen_pdf_bytes(df_res)
                if pdf_res:
                    st.download_button("📄 PDF Resumen", pdf_res, "Resumen.pdf", "application/pdf")

                # Préstamos
                df_prest = df[df["Clasificación"] == "Préstamos"]
                if not df_prest.empty:
                    st.subheader("Préstamos")
                    st.dataframe(df_prest)

                # Movimientos Limpios
                st.subheader("Movimientos")
                st.dataframe(df[["fecha_raw", "descripcion", "debito", "credito"]].style.format({"debito": "{:,.2f}", "credito": "{:,.2f}"}))

                # Excel
                sheets = {"Movimientos": df, "Resumen": df_res, "Prestamos": df_prest}
                st.download_button("📥 Descargar Excel", df_to_excel_bytes(sheets), "Credicoop_Full.xlsx")

        except Exception as e:
            st.error(f"Error técnico: {e}")
