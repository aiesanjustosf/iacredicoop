# ia_resumen_credicoop.py
# IA Resumen Credicoop (Cuenta Corriente Comercial PDF) - VERSIÓN CORREGIDA FINAL
# Regla estricta: 1 movimiento por línea. Si hay 2 montos, el derecho es saldo.
# Developer: Alfonso Alderete

import io
import re
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(
    page_title="IA Resumen Credicoop",
    layout="wide",
)

# Intentar importar librerías
try:
    import pdfplumber
    import xlsxwriter
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
except ImportError as e:
    st.error(f"Falta librería: {e}. Instalar: pip install pdfplumber xlsxwriter reportlab pandas")
    st.stop()

# ---------------- UTILIDADES ----------------

def parse_currency(text):
    """Convierte texto formato '1.234,56' a float, limpiando basura OCR."""
    if not text: return 0.0
    # Limpiar caracteres que no sean dígitos, comas o puntos (y menos)
    clean = re.sub(r'[^\d,.-]', '', text)
    
    # Manejo de negativos (ej: 1.000,00-)
    is_negative = "-" in clean
    clean = clean.replace("-", "")
    
    try:
        # Formato ES: 1.234,56 -> 1234.56
        if "," in clean:
            clean = clean.replace(".", "").replace(",", ".")
        val = float(clean)
        return -val if is_negative else val
    except ValueError:
        return 0.0

def fmt_ar(n):
    """Formato visual Argentino."""
    if pd.isna(n): return "0,00"
    return "{:,.2f}".format(n).replace(",", "X").replace(".", ",").replace("X", ".")

def clasificar_movimiento(desc):
    """Etiqueta el movimiento para el cuadro de gastos."""
    d = desc.upper()
    if "25413" in d or "25.413" in d: return "Impuesto Ley 25.413"
    if "SIRCREB" in d: return "SIRCREB"
    if "IVA" in d and ("PERCEPCION" in d or "RG" in d): return "Percepciones IVA"
    if "IVA" in d and "DEBITO FISCAL" in d: return "IVA Débito Fiscal"
    if "INTERES" in d: return "Intereses"
    if any(x in d for x in ["COMISION", "MANTEN", "SERV.", "GASTOS"]): return "Gastos/Comisiones"
    if "PRESTAMO" in d or "MUTUO" in d: return "Préstamo"
    return "Otros Operativos"

# ---------------- LÓGICA DE EXTRACCIÓN (CORE) ----------------

def extract_data_strict(pdf_file):
    """
    Lógica basada en columnas espaciales y regla de "Monto vs Saldo".
    """
    movements = []
    resumen_items = []
    
    saldo_inicial = 0.0
    saldo_final_pdf = 0.0
    fecha_cierre = ""
    
    # Punto de corte por defecto (se ajusta dinámicamente si encuentra encabezados)
    # Todo lo que esté a la izquierda de X=430 es Débito/Descripción
    # Todo lo que esté a la derecha de X=430 es Crédito (salvo que sea saldo)
    split_point_x = 420 
    
    # Rango donde suele estar el Saldo (muy a la derecha)
    saldo_column_start_x = 510

    capturando_resumen = False

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Extraer palabras con sus coordenadas
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=True)
            
            # 1. Intentar calibrar columnas si vemos encabezados
            header_deb = next((w for w in words if "DEBITO" in w['text'].upper()), None)
            header_cred = next((w for w in words if "CREDITO" in w['text'].upper()), None)
            
            if header_deb and header_cred:
                # El punto de corte es el medio entre el final de DEBITO y el inicio de CREDITO
                split_point_x = (header_deb['x1'] + header_cred['x0']) / 2
            
            # 2. Agrupar palabras por líneas (Eje Y)
            lines = {}
            for w in words:
                y = round(w['top']) # Redondear para agrupar palabras en la misma linea visual
                if y not in lines: lines[y] = []
                lines[y].append(w)
            
            # Ordenar líneas de arriba a abajo
            sorted_y = sorted(lines.keys())
            
            # Buffer para descripciones de varias líneas
            last_mov_idx = -1 
            
            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words]).strip()
                
                # --- DETECCIÓN DE ESTRUCTURA ---
                
                # Saldo Anterior (Inicio)
                if "SALDO ANTERIOR" in line_text.upper():
                    try:
                        # Buscar el último número de la línea
                        nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                        if nums: saldo_inicial = parse_currency(nums[-1]['text'])
                    except: pass
                    continue
                
                # Saldo Final (Fin de extracto)
                if "SALDO AL" in line_text.upper():
                    # Capturar fecha y activar modo resumen
                    parts = line_text.upper().split("SALDO AL")
                    if len(parts) > 1:
                        fecha_cierre = parts[1].strip().split(" ")[0]
                    
                    # Buscar el monto del saldo final (suele estar a la derecha)
                    nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                    if nums: saldo_final_pdf = parse_currency(nums[-1]['text'])
                    
                    capturando_resumen = True
                    continue
                
                # Modo Resumen Operativo (Tabla final)
                if capturando_resumen:
                    if "LIQUIDACION" in line_text.upper() or "DEBITO DIRECTO" in line_text.upper():
                        capturando_resumen = False
                    else:
                        # Si hay un concepto y un importe
                        nums = [w for w in line_words if re.match(r'^-?[\d\.]+,[\d]{2}$', w['text'])]
                        if nums:
                            monto = parse_currency(nums[-1]['text'])
                            # El texto es todo lo que no sea el número
                            concepto = " ".join([w['text'] for w in line_words if w != nums[-1]])
                            if len(concepto) > 3 and "CREDITO DE IMPUESTO" not in concepto.upper():
                                resumen_items.append({"Concepto": concepto, "Importe": monto})
                    continue

                # --- PROCESAMIENTO DE MOVIMIENTOS ---
                
                # Identificar si la línea empieza con fecha (Nuevo Movimiento)
                first_word = line_words[0]['text']
                is_new_movement = re.match(r'\d{2}/\d{2}/\d{2}', first_word)
                
                if is_new_movement:
                    fecha = first_word
                    
                    # Separar palabras que son MONTOS de palabras que son TEXTO
                    # Regex: busca formato numérico estricto (ej: 100,00 o 1.000,00)
                    money_tokens = []
                    text_tokens = []
                    comprobante = ""
                    
                    for w in line_words[1:]: # Saltar la fecha
                        txt = w['text']
                        # Es dinero?
                        if re.match(r'^-?[\d\.]+,[\d]{2}$', txt):
                            money_tokens.append(w)
                        # Es comprobante? (Solo números, largo > 4, sin comas)
                        elif re.match(r'^\d{5,}$', txt):
                            comprobante = txt
                        else:
                            text_tokens.append(txt)
                    
                    descripcion = " ".join(text_tokens)
                    
                    debito = 0.0
                    credito = 0.0
                    
                    # --- APLICACIÓN DE LA REGLA DEL USUARIO ---
                    # "Si aparecen dos montos, el de la derecha es el subtotal parcial"
                    
                    if len(money_tokens) > 0:
                        # Ordenar por posición X (izquierda a derecha)
                        money_tokens.sort(key=lambda w: w['x0'])
                        
                        # El monto de la transacción es SIEMPRE el primero (el más a la izquierda)
                        # Si hay un segundo monto (money_tokens[1]), es el Saldo, y lo ignoramos.
                        token_transaccion = money_tokens[0]
                        monto = parse_currency(token_transaccion['text'])
                        
                        # Ahora decidimos si es DEBITO o CREDITO según su posición X
                        x_pos = (token_transaccion['x0'] + token_transaccion['x1']) / 2
                        
                        if x_pos < split_point_x:
                            debito = monto
                        else:
                            credito = monto
                    
                    # Guardar movimiento
                    movements.append({
                        "Fecha": fecha,
                        "Comprobante": comprobante,
                        "Descripción": descripcion,
                        "Débito": debito,
                        "Crédito": credito
                    })
                    last_mov_idx = len(movements) - 1
                
                elif last_mov_idx >= 0 and not capturando_resumen:
                    # Línea de continuación (sin fecha, dentro de la tabla)
                    # Verificar que no sea un encabezado repetido
                    if "FECHA" not in line_text and "HOJA" not in line_text and "SALDO" not in line_text:
                        # Añadir texto a la descripción anterior
                        movements[last_mov_idx]["Descripción"] += " " + line_text

    return pd.DataFrame(movements), saldo_inicial, saldo_final_pdf, fecha_cierre, resumen_items

# ---------------- GENERACIÓN DE PDF Y EXCEL ----------------

def generar_pdf_resumen(data_resumen, fecha):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elems = []
    styles = getSampleStyleSheet()
    
    elems.append(Paragraph(f"Resumen Operativo (Conciliación al {fecha})", styles['Title']))
    elems.append(Spacer(1, 15))
    
    table_data = [["Concepto", "Importe"]]
    total = 0
    for item in data_resumen:
        table_data.append([item['Concepto'], fmt_ar(item['Importe'])])
        total += item['Importe']
    table_data.append(["TOTAL", fmt_ar(total)])
    
    t = Table(table_data, colWidths=[350, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.navy),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ]))
    elems.append(t)
    doc.build(elems)
    buffer.seek(0)
    return buffer

def generar_excel_completo(df_mov, df_res, df_prest):
    out = io.BytesIO()
    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    
    # Formatos
    workbook = writer.book
    fmt_num = workbook.add_format({'num_format': '#,##0.00'})
    
    # Sheet 1: Movimientos
    df_mov.to_excel(writer, sheet_name='Movimientos', index=False)
    ws = writer.sheets['Movimientos']
    ws.set_column('C:C', 50) # Descripcion ancha
    ws.set_column('D:E', 15, fmt_num) # Columnas moneda
    
    # Sheet 2: Resumen Operativo
    if not df_res.empty:
        df_res.to_excel(writer, sheet_name='Resumen Operativo', index=False)
        ws2 = writer.sheets['Resumen Operativo']
        ws2.set_column('A:A', 60)
        ws2.set_column('B:B', 15, fmt_num)
        
    # Sheet 3: Préstamos
    if not df_prest.empty:
        df_prest.to_excel(writer, sheet_name='Prestamos Bancarios', index=False)
        ws3 = writer.sheets['Prestamos Bancarios']
        ws3.set_column('C:C', 40)
        ws3.set_column('D:E', 15, fmt_num)
        
    writer.close()
    out.seek(0)
    return out

# ---------------- UI PRINCIPAL ----------------

st.title("🤖 Conciliador Bancario Credicoop")
st.markdown("""
Esta herramienta procesa el PDF detectando automáticamente:
1. **Movimientos:** Débitos a la izquierda, Créditos a la derecha (ignorando columna Saldo).
2. **Resumen Operativo:** Captura la tabla de impuestos al final.
3. **Préstamos:** Filtra operaciones de préstamos.
""")

uploaded_file = st.file_uploader("Subir PDF", type="pdf")

if uploaded_file:
    with st.spinner("Procesando estructura del PDF..."):
        df, s_ini, s_fin, fecha, lista_resumen = extract_data_strict(uploaded_file)
    
    if df.empty:
        st.error("No se encontraron movimientos válidos.")
    else:
        # Cálculos de conciliación
        tot_deb = df['Débito'].sum()
        tot_cred = df['Crédito'].sum()
        saldo_calc = s_ini + tot_cred - tot_deb
        diff = saldo_calc - s_fin
        
        # --- METRICAS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Saldo Inicial", fmt_ar(s_ini))
        col2.metric("Total Débitos", fmt_ar(tot_deb), delta_color="inverse")
        col3.metric("Total Créditos", fmt_ar(tot_cred))
        col4.metric("Diferencia Conciliación", fmt_ar(diff), 
                    delta="OK" if abs(diff) < 1 else "ERROR",
                    delta_color="normal" if abs(diff) < 1 else "inverse")
        
        if abs(diff) > 1:
            st.warning(f"Atención: El saldo calculado (${fmt_ar(saldo_calc)}) difiere del saldo final del PDF (${fmt_ar(s_fin)}). Revisar saltos de página.")
        else:
            st.success("Conciliación Exitosa.")

        # --- PREPARAR DATOS ---
        # 1. Clasificar movimientos
        df['Clasificación'] = df['Descripción'].apply(clasificar_movimiento)
        
        # 2. DataFrame Resumen
        if lista_resumen:
            df_resumen = pd.DataFrame(lista_resumen)
        else:
            st.info("Generando resumen operativo desde los movimientos (tabla final no detectada).")
            resumen_agrupado = df[df['Débito'] > 0].groupby('Clasificación')['Débito'].sum().reset_index()
            resumen_agrupado.columns = ['Concepto', 'Importe']
            df_resumen = resumen_agrupado
        
        # 3. DataFrame Préstamos
        df_prestamos = df[df['Clasificación'] == 'Préstamo'].copy()

        # --- PESTAÑAS ---
        tab_mov, tab_res, tab_prest = st.tabs(["Movimientos", "Resumen Operativo", "Préstamos"])
        
        with tab_mov:
            st.dataframe(df.style.format({"Débito": "{:,.2f}", "Crédito": "{:,.2f}"}))
            
        with tab_res:
            st.dataframe(df_resumen.style.format({"Importe": "{:,.2f}"}))
            # Botón PDF Resumen
            pdf_data = generar_pdf_resumen(df_resumen.to_dict('records'), fecha)
            st.download_button("📄 Descargar PDF Resumen Operativo", data=pdf_data, file_name="Resumen_Operativo.pdf", mime="application/pdf")
            
        with tab_prest:
            if df_prestamos.empty:
                st.write("No hay préstamos.")
            else:
                st.dataframe(df_prestamos)
                
        # --- EXCEL FINAL ---
        st.markdown("---")
        excel_bytes = generar_excel_completo(df, df_resumen, df_prestamos)
        st.download_button(
            "📥 Descargar Excel Completo (Todas las hojas)",
            data=excel_bytes,
            file_name="Conciliacion_Credicoop.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
