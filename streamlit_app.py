# streamlit_app.py
# Herramienta para uso interno - AIE San Justo
# VERSIÓN: COLUMN MAPPING (Mapeo estricto por encabezados)

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
    layout="wide",
)

st.markdown("""
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      h1 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# ---------------- LIBRERÍAS ----------------
try:
    import pdfplumber
    import xlsxwriter
except ImportError as e:
    st.error(f"Falta librería: {e}")
    st.stop()

# ---------------- ASSETS ----------------
HERE = Path(__file__).parent
LOGO = HERE / "logo_aie.png"
if LOGO.exists():
    st.image(str(LOGO), width=200)

st.title("IA Resumen Credicoop (Motor de Conciliación)")

# ---------------- FUNCIONES AUXILIARES ----------------

def parse_currency(text):
    """Convierte texto (1.234,56) a float."""
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

# ---------------- LÓGICA DE EXTRACCIÓN (MAPEO DE COLUMNAS) ----------------

def extract_data_strict_columns(pdf_bytes, filename):
    movements = []
    
    saldo_inicial = 0.0
    saldo_final_pdf = 0.0
    fecha_cierre = ""
    
    # Variables para coordenadas de columnas (se calibran con los encabezados)
    x_debito_start = 0
    x_credito_start = 0
    x_saldo_start = 0 # Para ignorar lo que esté a la derecha
    
    # Banderas de estado
    encontrado_saldo_inicial = False
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=True)
            
            # 1. BUSCAR ENCABEZADOS PARA CALIBRAR COLUMNAS
            # Buscamos la fila que tiene los títulos
            header_deb = next((w for w in words if "DEBITO" in w['text'].upper()), None)
            header_cred = next((w for w in words if "CREDITO" in w['text'].upper()), None)
            header_saldo = next((w for w in words if "SALDO" in w['text'].upper() and w['x0'] > (page.width/2)), None) # Saldo a la derecha
            
            # Definir límite vertical (Y) para ignorar logos y encabezados superiores
            y_start_content = 0
            
            if header_deb and header_cred:
                # Calibración precisa basada en esta página
                x_debito_start = header_deb['x0'] - 20 # Margen izq
                x_credito_start = header_cred['x0'] - 10
                y_start_content = header_deb['bottom'] + 5 # Empezar a leer abajo de los titulos
                
                if header_saldo:
                    x_saldo_start = header_saldo['x0'] - 10
                else:
                    x_saldo_start = page.width - 80 # Default borde derecho
            elif page.page_number == 1:
                # Si no encuentra headers en pag 1, usar defaults aproximados
                x_debito_start = 380
                x_credito_start = 450
                x_saldo_start = 520
                y_start_content = 150
            else:
                # Páginas siguientes sin header (usa calibración anterior)
                y_start_content = 50 

            # 2. FILTRAR Y AGRUPAR LINEAS
            content_words = [w for w in words if w['top'] > y_start_content]
            
            lines = {}
            for w in content_words:
                y = round(w['top'])
                if y not in lines: lines[y] = []
                lines[y].append(w)
            
            sorted_y = sorted(lines.keys())
            last_mov_idx = -1 # Para concatenar descripciones

            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words]).strip()
                line_text_upper = line_text.upper()

                # Ignorar líneas de paginación
                if "CONTINUA EN PAGINA" in line_text_upper or "VIENE DE PAGINA" in line_text_upper:
                    continue

                # --- A. DETECCIÓN DE SALDO ANTERIOR ---
                # (No tiene fecha, solo dice SALDO ANTERIOR y un monto en la columna Saldo)
                if "SALDO" in line_text_upper and "ANTERIOR" in line_text_upper:
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums:
                        # El saldo anterior suele estar en la última columna
                        saldo_inicial = parse_currency(nums[-1]['text'])
                        encontrado_saldo_inicial = True
                    continue

                # --- B. DETECCIÓN DE SALDO FINAL (Cierre) ---
                if "SALDO AL" in line_text_upper:
                    parts = line_text_upper.split("SALDO AL")
                    if len(parts) > 1:
                        fecha_cierre = parts[1].strip().split(" ")[0]
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums:
                        saldo_final_pdf = parse_currency(nums[-1]['text'])
                    # Aquí termina el procesamiento de movimientos
                    break 

                # --- C. PROCESAMIENTO DE MOVIMIENTOS ---
                # Un movimiento inicia con Fecha (dd/mm/yy)
                date_match = re.search(r'^\s*\d{2}/\d{2}/\d{2}', line_text)
                
                if date_match:
                    fecha_raw = date_match.group(0).strip()
                    try:
                        dt = datetime.strptime(fecha_raw, "%d/%m/%y").date()
                    except:
                        dt = None

                    # Separar tokens de texto y números
                    desc_parts = []
                    comprobante = ""
                    debito = 0.0
                    credito = 0.0
                    
                    for w in line_words:
                        txt = w['text']
                        x_center = (w['x0'] + w['x1']) / 2
                        
                        # Es parte de la fecha?
                        if txt in fecha_raw: continue
                        
                        # Es un número (monto)?
                        if re.match(r'^-?[\d\.]+,[\d]{2}$', txt):
                            val = parse_currency(txt)
                            
                            # CLASIFICACIÓN POR COLUMNA (GEOMETRÍA ESTRICTA)
                            if x_center >= x_debito_start and x_center < x_credito_start:
                                debito = val
                            elif x_center >= x_credito_start and x_center < x_saldo_start:
                                credito = val
                            # Si es > x_saldo_start, es el saldo parcial, lo ignoramos.
                        
                        # Es comprobante? (Números enteros largos a la izquierda)
                        elif re.match(r'^\d{4,}$', txt) and "," not in txt and x_center < x_debito_start:
                            comprobante = txt
                        
                        # Es descripción?
                        else:
                            # Evitar agregar números sueltos o partes de fecha
                            if not re.match(r'^-?[\d\.]+,[\d]{2}$', txt): 
                                desc_parts.append(txt)
                    
                    descripcion = " ".join(desc_parts).strip()
                    
                    movements.append({
                        "fecha": dt,
                        "fecha_raw": fecha_raw,
                        "comprobante": comprobante,
                        "descripcion": descripcion,
                        "debito": debito,
                        "credito": credito
                    })
                    last_mov_idx = len(movements) - 1

                # --- D. LINEAS DE CONTINUACIÓN (Descripción Multilínea) ---
                elif last_mov_idx >= 0 and encontrado_saldo_inicial:
                    # Si no es fecha, ni saldo, ni header repetido, es continuación de descripción
                    if "FECHA" not in line_text_upper and "SALDO" not in line_text_upper:
                         # Solo agregar texto, ignorar montos que aparezcan (suelen ser saldos transportados o basura)
                         text_only = " ".join([w['text'] for w in line_words if not re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])])
                         movements[last_mov_idx]["descripcion"] += " " + text_only

    meta = {
        "saldo_inicial": saldo_inicial,
        "saldo_final": saldo_final_pdf,
        "fecha_cierre": fecha_cierre
    }
    return pd.DataFrame(movements), meta

# ---------------- GENERADORES EXCEL ----------------

def generate_excel(df, meta):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        # HOJA 1: Conciliación y Movimientos
        wb = writer.book
        ws = wb.add_worksheet("Conciliacion")
        writer.sheets["Conciliacion"] = ws
        
        fmt_bold = wb.add_format({'bold': True})
        fmt_money = wb.add_format({'num_format': '#,##0.00'})
        fmt_date = wb.add_format({'num_format': 'dd/mm/yyyy'})
        fmt_header = wb.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
        
        # Encabezado Conciliación
        ws.write(0, 0, "CONCILIACIÓN BANCARIA", fmt_bold)
        ws.write(2, 0, "Saldo Inicial (Extracto):", fmt_bold)
        ws.write(2, 1, meta["saldo_inicial"], fmt_money)
        
        tot_cred = df["credito"].sum()
        tot_deb = df["debito"].sum()
        saldo_calc = meta["saldo_inicial"] + tot_cred - tot_deb
        
        ws.write(3, 0, "(+) Créditos:", fmt_bold)
        ws.write(3, 1, tot_cred, fmt_money)
        ws.write(4, 0, "(-) Débitos:", fmt_bold)
        ws.write(4, 1, tot_deb, fmt_money)
        ws.write(5, 0, "(=) Saldo Calculado:", fmt_bold)
        ws.write(5, 1, saldo_calc, fmt_money)
        
        ws.write(2, 3, "Saldo Final (Extracto):", fmt_bold)
        ws.write(2, 4, meta["saldo_final"], fmt_money)
        ws.write(3, 3, "Diferencia:", fmt_bold)
        ws.write(3, 4, saldo_calc - meta["saldo_final"], fmt_money)
        
        # Tabla Movimientos
        start_row = 8
        headers = ["Fecha", "Comprobante", "Descripción", "Débito", "Crédito"]
        for col, h in enumerate(headers):
            ws.write(start_row, col, h, fmt_header)
            
        for i, row in df.iterrows():
            r = start_row + 1 + i
            ws.write(r, 0, row["fecha"], fmt_date)
            ws.write(r, 1, row["comprobante"])
            ws.write(r, 2, row["descripcion"])
            ws.write(r, 3, row["debito"], fmt_money)
            ws.write(r, 4, row["credito"], fmt_money)
            
        ws.set_column(0, 0, 12)
        ws.set_column(1, 1, 15)
        ws.set_column(2, 2, 50) # Descripcion ancha
        ws.set_column(3, 4, 15)

    return out.getvalue()

# ---------------- APP PRINCIPAL ----------------

uploaded = st.file_uploader("Subí el PDF del Resumen Credicoop", type=["pdf"])

if uploaded is not None:
    with st.spinner("Leyendo y conciliando..."):
        try:
            pdf_bytes = uploaded.read()
            df, meta = extract_data_strict_columns(pdf_bytes, uploaded.name)
            
            if df.empty:
                st.error("No se encontraron movimientos. ¿Es el PDF correcto?")
            else:
                # Cálculos
                t_deb = df["debito"].sum()
                t_cred = df["credito"].sum()
                s_ini = meta["saldo_inicial"]
                s_fin = meta["saldo_final"]
                s_calc = s_ini + t_cred - t_deb
                diff = s_calc - s_fin
                
                # --- VISUALIZACIÓN ---
                
                # 1. Panel de Conciliación
                st.subheader("Estado de Conciliación")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Saldo Inicial", fmt_ar(s_ini))
                c2.metric("Créditos (+)", fmt_ar(t_cred))
                c3.metric("Débitos (-)", fmt_ar(t_deb))
                
                concil_status = "OK ✅" if abs(diff) < 1.0 else "DIFERENCIA ❌"
                c4.metric("Diferencia", fmt_ar(diff), delta=concil_status, delta_color="normal" if abs(diff)<1 else "inverse")
                
                if abs(diff) > 1.0:
                    st.warning(f"Saldo Calculado: {fmt_ar(s_calc)} | Saldo PDF: {fmt_ar(s_fin)}")
                
                # 2. Grilla
                st.markdown("### Detalle de Movimientos")
                st.dataframe(
                    df[["fecha_raw", "comprobante", "descripcion", "debito", "credito"]].style.format({"debito": "{:,.2f}", "credito": "{:,.2f}"}),
                    use_container_width=True,
                    height=500
                )
                
                # 3. Descarga
                excel_data = generate_excel(df, meta)
                st.download_button(
                    label="📥 Descargar Excel Conciliado",
                    data=excel_data,
                    file_name="Conciliacion_Credicoop.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
