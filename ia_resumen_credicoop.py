# ia_resumen_credicoop.py
# Herramienta para uso interno - AIE San Justo
# Developer: Alfonso Alderete
# VERSIÓN: Spatial Strict (Estética Original + Lógica Corregida)

import io
import re
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------- CONFIGURACIÓN E IMPORTACIONES ----------------
try:
    import pdfplumber
    import xlsxwriter
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_OK = True
except ImportError as e:
    st.error(f"Error de importación: {e}. Revisá requirements.txt")
    st.stop()

# ---------------- UI / ASSETS (Estética Original) ----------------
HERE = Path(__file__).parent
LOGO = HERE / "logo_aie.png"
FAVICON = HERE / "favicon-aie.ico"

st.set_page_config(
    page_title="IA Resumen Credicoop",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="centered", # Respetando tu layout original
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

# ---------------- UTILIDADES ----------------

def parse_currency(text):
    """Convierte texto formato '1.234,56' a float."""
    if not text: return 0.0
    # Limpiar todo lo que no sea número, coma, punto o signo menos
    clean = re.sub(r'[^\d,.-]', '', str(text))
    # Manejo de negativos al final (ej: 1.000,00-)
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
    """Formato visual para la UI (con separador de miles)."""
    if pd.isna(n) or n is None: return "—"
    return "{:,.2f}".format(n).replace(",", "X").replace(".", ",").replace("X", ".")

def clasificar(desc):
    """Clasifica el movimiento para el resumen operativo."""
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

# ---------------- LÓGICA DE EXTRACCIÓN (CORE SPATIAL) ----------------

def extract_data_pdf(pdf_bytes, filename):
    movements = []
    resumen_items = []
    saldo_inicial = np.nan
    saldo_final_pdf = np.nan
    fecha_cierre = ""
    
    # Lógica espacial: Si la coordenada X del monto es menor a esto, es DEBITO.
    # Se ajusta solo si encuentra encabezados. Valor default seguro para A4.
    split_x_default = 420 
    
    capturando_resumen = False

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # Extraer palabras con sus coordenadas (x0, top, etc.)
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=True)
            
            # 1. Calibración dinámica de columnas
            # Buscamos dónde terminan los DEBITOS y empiezan los CREDITOS
            header_deb = next((w for w in words if "DEBITO" in w['text'].upper()), None)
            header_cred = next((w for w in words if "CREDITO" in w['text'].upper()), None)
            
            if header_deb and header_cred:
                # El corte es el promedio entre el fin de uno y el inicio del otro
                split_point = (header_deb['x1'] + header_cred['x0']) / 2
            else:
                split_point = split_x_default

            # 2. Agrupar palabras por líneas visuales (eje Y)
            lines = {}
            for w in words:
                y = round(w['top']) # Redondear para agrupar
                if y not in lines: lines[y] = []
                lines[y].append(w)
            
            sorted_y = sorted(lines.keys())
            
            last_idx = -1 # Para concatenar descripciones multilínea

            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                # Reconstruir texto de la línea para búsqueda de patrones
                line_text = " ".join([w['text'] for w in line_words]).strip()
                line_text_upper = line_text.upper()

                # --- A. DETECCIÓN DE SALDOS GLOBALES ---
                if "SALDO ANTERIOR" in line_text_upper:
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_inicial = parse_currency(nums[-1]['text'])
                    continue
                
                if "SALDO AL" in line_text_upper:
                    # Capturar fecha de cierre
                    parts = line_text_upper.split("SALDO AL")
                    if len(parts) > 1:
                        fecha_cierre = parts[1].strip().split(" ")[0]
                    # Capturar saldo final
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_final_pdf = parse_currency(nums[-1]['text'])
                    capturando_resumen = True # Activar lectura de tabla final
                    continue

                # --- B. LECTURA DE RESUMEN OPERATIVO (Final del extracto) ---
                if capturando_resumen:
                    # Criterios de parada
                    if "LIQUIDACION" in line_text_upper or "DEBITO DIRECTO" in line_text_upper:
                        capturando_resumen = False
                    else:
                        # Buscar patrón: Texto de concepto ..... Monto
                        nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                        if nums:
                            # Asumimos que el monto es el último número de la línea
                            monto_obj = nums[-1]
                            monto = parse_currency(monto_obj['text'])
                            # El concepto es todo lo demás
                            concepto = " ".join([w['text'] for w in line_words if w != monto_obj])
                            
                            # Filtros anti-ruido
                            if len(concepto) > 3 and "CREDITO DE IMPUESTO" not in concepto.upper():
                                resumen_items.append({"Concepto": concepto, "Importe": monto})
                    continue

                # --- C. LECTURA DE MOVIMIENTOS (Cuerpo principal) ---
                
                # Detectar si es una línea nueva de movimiento (Empieza con Fecha)
                first_word = line_words[0]['text']
                is_date = re.match(r'\d{2}/\d{2}/\d{2}', first_word) # dd/mm/yy o dd/mm/yyyy

                if is_date:
                    fecha_raw = first_word
                    try:
                        fecha_dt = datetime.strptime(fecha_raw[:8], "%d/%m/%y").date()
                    except:
                        try:
                            fecha_dt = datetime.strptime(fecha_raw[:10], "%d/%m/%Y").date()
                        except:
                            fecha_dt = None

                    # Separar tokens de dinero y tokens de texto
                    money_tokens = []
                    text_tokens = []
                    comprobante = ""
                    
                    for w in line_words[1:]: # Saltar la fecha
                        txt = w['text']
                        # Regex estricto de moneda (requiere coma decimal)
                        if re.match(r'^-?[\d\.]+,[\d]{2}$', txt):
                            money_tokens.append(w)
                        elif re.match(r'^\d{4,}$', txt) and "," not in txt: # Comprobante (nro largo)
                            comprobante = txt
                        else:
                            text_tokens.append(txt)
                    
                    descripcion = " ".join(text_tokens)
                    debito = 0.0
                    credito = 0.0
                    
                    # --- REGLA DE ORO ---
                    # Si hay montos, el de la derecha puede ser saldo.
                    # El de la izquierda es el movimiento.
                    if money_tokens:
                        # Ordenar visualmente izquierda a derecha
                        money_tokens.sort(key=lambda w: w['x0'])
                        
                        # Tomamos el PRIMERO (el más a la izquierda) como el monto de operación
                        op_token = money_tokens[0]
                        valor = parse_currency(op_token['text'])
                        
                        # Decidimos si es DEBITO o CREDITO según coordenada X
                        center_x = (op_token['x0'] + op_token['x1']) / 2
                        
                        if center_x < split_point:
                            debito = valor
                        else:
                            credito = valor
                            
                    movements.append({
                        "fecha": fecha_dt,
                        "fecha_raw": fecha_raw,
                        "comprobante": comprobante,
                        "descripcion": descripcion,
                        "debito": debito,
                        "credito": credito,
                        "archivo": filename,
                        "pagina": page.page_number
                    })
                    last_idx = len(movements) - 1

                elif last_idx >= 0 and not capturando_resumen:
                    # Línea de continuación (sin fecha)
                    # Evitar encabezados de tabla repetidos en nuevas páginas
                    if "FECHA" not in line_text_upper and "SALDO" not in line_text_upper:
                         movements[last_idx]["descripcion"] += " " + line_text

    meta = {
        "saldo_inicial": saldo_inicial,
        "saldo_final": saldo_final_pdf,
        "fecha_final": fecha_cierre
    }
    
    return pd.DataFrame(movements), meta, resumen_items

# ---------------- GENERADORES DE ARCHIVOS ----------------

def df_to_excel_bytes(sheets):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_money = wb.add_format({"num_format": "#,##0.00"})
        fmt_date = wb.add_format({"num_format": "dd/mm/yyyy"})

        for name, df in sheets.items():
            if df.empty: continue
            df.to_excel(writer, index=False, sheet_name=name[:30])
            ws = writer.sheets[name[:30]]
            
            # Ajuste ancho columnas
            for idx, col in enumerate(df.columns):
                ws.set_column(idx, idx, 15)
            
            # Formatos específicos
            if "descripcion" in df.columns:
                idx = df.columns.get_loc("descripcion")
                ws.set_column(idx, idx, 50)
            if "debito" in df.columns:
                idx = df.columns.get_loc("debito")
                ws.set_column(idx, idx, 18, fmt_money)
            if "credito" in df.columns:
                idx = df.columns.get_loc("credito")
                ws.set_column(idx, idx, 18, fmt_money)
            if "Importe" in df.columns: # Para resumen operativo
                idx = df.columns.get_loc("Importe")
                ws.set_column(idx, idx, 18, fmt_money)
            if "fecha" in df.columns:
                idx = df.columns.get_loc("fecha")
                ws.set_column(idx, idx, 12, fmt_date)
                
    return out.getvalue()

def resumen_pdf_bytes(df_res):
    if df_res.empty: return None
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title="Resumen Operativo")
    styles = getSampleStyleSheet()
    elems = [
        Paragraph("Resumen Operativo: Registración Módulo IVA", styles["Title"]),
        Spacer(1, 10)
    ]
    
    data = [["Concepto", "Importe"]] + [[r["Concepto"], fmt_ar(r["Importe"])] for r in df_res.to_dict('records')]
    
    # Calcular total para el PDF
    total = df_res["Importe"].sum()
    data.append(["TOTAL", fmt_ar(total)])

    tbl = Table(data, colWidths=[340, 160])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"), # Total en negrita
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("AIE San Justo - Credicoop Parser", styles["Normal"]))
    doc.build(elems)
    return pdf_buf.getvalue()

# ---------------- APP PRINCIPAL ----------------

uploaded = st.file_uploader("Subí un PDF del resumen bancario (Banco Credicoop)", type=["pdf"])

if uploaded is not None:
    # Procesamiento
    with st.spinner("Procesando PDF..."):
        pdf_bytes = uploaded.read()
        df, meta, lista_resumen = extract_data_pdf(pdf_bytes, uploaded.name)

    if df.empty:
        st.error("⚠️ El PDF se procesó pero no se encontraron movimientos. Verificá que sea un archivo 'Resumen de Cuenta' original y no una imagen escaneada.")
    else:
        # Cálculos
        df["Clasificación"] = df["descripcion"].apply(clasificar)
        
        total_deb = df["debito"].sum()
        total_cred = df["credito"].sum()
        saldo_ini = meta["saldo_inicial"]
        saldo_fin_pdf = meta["saldo_final"]
        
        # Conciliación
        saldo_calc = np.nan
        diff = np.nan
        if not np.isnan(saldo_ini):
            saldo_calc = saldo_ini + total_cred - total_deb
            if not np.isnan(saldo_fin_pdf):
                diff = saldo_calc - saldo_fin_pdf

        # --- MOSTRAR DATOS (Diseño Original) ---
        
        # 1. Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Saldo Anterior", fmt_ar(saldo_ini))
        c2.metric("Total Créditos (+)", fmt_ar(total_cred))
        c3.metric("Total Débitos (–)", fmt_ar(total_deb))

        c4, c5, c6 = st.columns(3)
        c4.metric("Saldo PDF (Fin)", fmt_ar(saldo_fin_pdf))
        c5.metric("Saldo Calculado", fmt_ar(saldo_calc))
        
        if not np.isnan(diff):
            color = "normal" if abs(diff) < 1.0 else "inverse"
            c6.metric("Diferencia", fmt_ar(diff), delta="Conciliado" if abs(diff)<1 else "Error", delta_color=color)

        # 2. Resumen Operativo
        st.markdown("---")
        st.caption("Resumen Operativo: Registración Módulo IVA")
        
        # Si extrajimos la tabla del PDF, la usamos. Si no, la calculamos.
        if lista_resumen:
            df_res = pd.DataFrame(lista_resumen)
            st.info("ℹ️ Tabla de impuestos extraída directamente del PDF.")
        else:
            st.warning("⚠️ No se detectó la tabla final en el PDF. Calculando estimación basada en movimientos.")
            # Agrupar solo débitos
            df_res = df[df["debito"] > 0].groupby("Clasificación")["debito"].sum().reset_index()
            df_res.rename(columns={"debito": "Importe", "Clasificación": "Concepto"}, inplace=True)

        # Mostrar tabla resumen
        st.dataframe(df_res.style.format({"Importe": "{:,.2f}"}), use_container_width=True, hide_index=True)

        # Descarga PDF Resumen
        pdf_res_bytes = resumen_pdf_bytes(df_res)
        if pdf_res_bytes:
            st.download_button(
                "📄 Descargar PDF – Resumen Operativo",
                data=pdf_res_bytes,
                file_name="Resumen_Operativo_Credicoop.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # 3. Préstamos
        st.caption("Detalle de préstamos")
        df_prest = df[df["Clasificación"] == "Préstamos"].copy()
        if not df_prest.empty:
            st.dataframe(df_prest[["fecha_raw", "descripcion", "debito", "credito"]], use_container_width=True)
        else:
            st.info("No se detectaron movimientos de préstamos.")

        # 4. Grilla Principal
        st.caption("Detalle de movimientos")
        st.dataframe(
            df[["fecha_raw", "comprobante", "descripcion", "Clasificación", "debito", "credito"]].style.format({"debito": "{:,.2f}", "credito": "{:,.2f}"}), 
            use_container_width=True, 
            hide_index=True
        )

        # 5. Descarga Excel Global
        sheets = {
            "Movimientos": df[["fecha", "fecha_raw", "comprobante", "descripcion", "Clasificación", "debito", "credito"]],
            "Resumen_Operativo": df_res,
            "Prestamos": df_prest
        }
        xlsx_bytes = df_to_excel_bytes(sheets)
        
        st.markdown("---")
        st.download_button(
            "📥 Descargar Excel Completo",
            data=xlsx_bytes,
            file_name="credicoop_completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

st.markdown("---")
st.caption("Herramienta para uso interno AIE San Justo | Developer Alfonso Alderete")
