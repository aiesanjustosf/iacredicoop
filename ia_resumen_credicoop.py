import io
import re
import pandas as pd
import streamlit as st
import pdfplumber
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --- CONFIGURACIÓN UI ---
st.set_page_config(page_title="IA Resumen Credicoop - Conciliación Espacial", layout="wide")
st.title("🏦 IA Resumen Credicoop: Conciliación Estricta")
st.markdown("""
**Corrección aplicada:** Se utiliza detección espacial de coordenadas X. 
Los montos se asignan a Debe/Haber según su posición física bajo los encabezados, solucionando el error de columnas invertidas.
""")

# --- UTILIDADES ---
def fmt_ar(n):
    if n is None or np.isnan(n): return "—"
    return f"$ {n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def parse_money(text):
    """Convierte texto 1.000,00 a float 1000.0"""
    if not text: return 0.0
    # Limpieza agresiva de caracteres invisibles
    text = str(text).replace(" ", "").replace("−", "-").replace("–", "-")
    # Formato argentino: quitar puntos de miles, cambiar coma decimal por punto
    clean = text.replace(".", "").replace(",", ".")
    try:
        return float(clean)
    except ValueError:
        return 0.0

def clasificar_movimiento(desc):
    """Clasifica el gasto para el resumen impositivo/operativo"""
    u = desc.upper()
    if "25413" in u or "25.413" in u: return "Ley 25.413 (Imp. Cheque)"
    if "SIRCREB" in u: return "SIRCREB"
    if "PERCEP" in u and "IVA" in u: return "Percepciones IVA"
    if "I.V.A." in u or "IVA " in u:
        if "10,5" in u: return "IVA 10,5%"
        return "IVA 21%"
    if any(x in u for x in ["COMISION", "MANTENIMIENTO", "SERVICIO", "GASTOS"]): return "Gastos Bancarios (Neto)"
    if any(x in u for x in ["PRESTAMO", "CUOTA", "AMORTIZACION"]): return "Préstamos"
    return "Otros Movimientos"

# --- LÓGICA CORE: PARSER ESPACIAL ---
def obtener_limites_columnas(pdf):
    """
    Busca en la primera página la posición X de los encabezados 'DEBITO' y 'CREDITO'
    para saber dónde cortar las columnas dinámicamente.
    """
    first_page = pdf.pages[0]
    words = first_page.extract_words()
    
    # Valores por defecto (por si falla la detección, basados en A4 estándar Credicoop)
    # Debito suele empezar ~380 y terminar ~450. Credito ~450 a ~520.
    limites = {
        "debito_start": 350,
        "debito_end": 460, # Frontera entre debito y credito
        "credito_end": 530 # Frontera entre credito y saldo
    }
    
    # Buscar coordenadas reales
    deb_box = None
    cred_box = None
    
    for w in words:
        if w['text'] == "DEBITO": deb_box = w
        if w['text'] == "CREDITO": cred_box = w
    
    if deb_box and cred_box:
        # El límite entre débito y crédito es el punto medio entre el fin de uno y el inicio del otro
        limites["debito_start"] = deb_box['x0'] - 20
        limites["debito_end"] = (deb_box['x1'] + cred_box['x0']) / 2
        limites["credito_end"] = cred_box['x1'] + 40 # Un margen a la derecha del crédito
        
    return limites

def procesar_pdf_espacial(pdf_bytes):
    rows = []
    saldo_inicial = 0.0
    saldo_final = 0.0
    resumen_impositivo_oficial = [] # Para guardar el cuadro del final del PDF
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # 1. Detectar Geometría
        limites = obtener_limites_columnas(pdf)
        
        full_text_for_footer = "" # Texto plano para buscar saldos finales y cuadros
        
        for page in pdf.pages:
            # Texto plano para búsquedas globales (Saldo final, cuadros)
            full_text_for_footer += page.extract_text() + "\n"
            
            # Extraer palabras con sus posiciones
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            
            # Agrupar palabras por líneas (mismo eje Y aprox)
            lines = {}
            for w in words:
                top = round(w['top']) # Redondear para agrupar líneas imperfectas
                if top not in lines: lines[top] = []
                lines[top].append(w)
            
            # Ordenar líneas de arriba a abajo
            sorted_y = sorted(lines.keys())
            
            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words])
                
                # Detectar si es una línea de movimiento (empieza con Fecha)
                match_date = re.match(r"^(\d{2}/\d{2}/\d{2,4})", line_text)
                
                if match_date:
                    fecha = match_date.group(1)
                    
                    # Inicializar
                    debito = 0.0
                    credito = 0.0
                    descripcion_parts = []
                    
                    # Analizar cada palabra de la línea
                    for w in line_words:
                        text = w['text']
                        x_center = (w['x0'] + w['x1']) / 2
                        
                        # Si parece un número monetario (tiene coma y formato numérico)
                        # Regex flexible para detectar números como 28.100,00 o 573,00
                        if re.match(r"^-?[\d\.]+,[\d]{2}$", text):
                            valor = parse_money(text)
                            
                            # CLASIFICACIÓN ESPACIAL (LA SOLUCIÓN)
                            if limites["debito_start"] < x_center < limites["debito_end"]:
                                debito = valor
                            elif limites["debito_end"] < x_center < limites["credito_end"]:
                                credito = valor
                            # Si está más a la derecha, es SALDO (lo ignoramos para el cálculo)
                        else:
                            # Si no es fecha ni el comprobante (asumimos comprobante es numérico corto al inicio)
                            if text != fecha and not (text.isdigit() and len(text) > 4 and line_words.index(w) < 2):
                                descripcion_parts.append(text)
                    
                    desc_final = " ".join(descripcion_parts)
                    
                    rows.append({
                        "Fecha": fecha,
                        "Descripción": desc_final,
                        "Débito": debito,
                        "Crédito": credito,
                        "Clasificación": clasificar_movimiento(desc_final)
                    })
                
                # Detectar Saldo Anterior (Suele estar en la primera línea de movimientos o cabecera)
                elif "SALDO" in line_text and "ANTERIOR" in line_text:
                    # Buscar el número que esté más a la derecha (columna saldo)
                    try:
                        numeros = [w['text'] for w in line_words if re.match(r"-?[\d\.]+,[\d]{2}", w['text'])]
                        if numeros:
                            saldo_inicial = parse_money(numeros[-1])
                    except:
                        pass

        # 2. Búsqueda de Saldo Final y Cuadro de Impuestos (Regex en texto completo)
        # Buscar Saldo al 30/11/xx
        match_saldo_fin = re.search(r"SALDO AL \d{2}/\d{2}/\d{2,4}\s+([\d\.,\-]+)", full_text_for_footer)
        if match_saldo_fin:
            saldo_final = parse_money(match_saldo_fin.group(1))
            
        # Extraer cuadro resumen oficial del banco (Si existe al final)
        # Esto extrae lo que dice "TOTAL IMPUESTO LEY 25413... X.XXX,XX"
        regex_taxes = [
            (r"TOTAL IMPUESTO LEY 25413.*?([\d\.,]+)", "Ley 25.413 (Total)"),
            (r"IVA ALIC ADIC RG 2408.*?([\d\.,]+)", "Percepción IVA RG 2408"),
            (r"IVA.*?ALICUOTA INSCRIPTO\s+PERCIBIDO.*?([\d\.,]+)", "IVA 21%"),
            (r"IVA.*?ALICUOTA INSCRIPTO REDUCIDA\s+PERCIBIDO.*?([\d\.,]+)", "IVA 10.5%")
        ]
        
        for reg, name in regex_taxes:
            match = re.search(reg, full_text_for_footer, re.DOTALL)
            if match:
                monto = parse_money(match.group(1))
                if monto > 0:
                    resumen_impositivo_oficial.append({"Concepto": name, "Importe": monto})

    return pd.DataFrame(rows), saldo_inicial, saldo_final, resumen_impositivo_oficial

# --- GENERADOR PDF (ReportLab) ---
def generar_pdf_reporte(df, resumen_imp, metricas):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Título
    elements.append(Paragraph("Informe de Conciliación Bancaria - Credicoop", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Métricas
    elements.append(Paragraph("Resumen General", styles['Heading2']))
    data_metrics = [
        ["Saldo Anterior", metricas['s_ant']],
        ["Total Créditos", metricas['t_cred']],
        ["Total Débitos", metricas['t_deb']],
        ["Saldo Calculado", metricas['s_calc']],
        ["Saldo Resumen", metricas['s_fin']],
        ["Diferencia", metricas['diff']]
    ]
    t_met = Table(data_metrics, colWidths=[200, 150])
    t_met.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)
    ]))
    elements.append(t_met)
    
    # Resumen Impositivo
    if resumen_imp:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Cuadro de Impuestos (Extracto Oficial)", styles['Heading2']))
        data_imp = [["Concepto", "Importe"]] + [[r['Concepto'], fmt_ar(r['Importe'])] for r in resumen_imp]
        t_imp = Table(data_imp, colWidths=[300, 100])
        t_imp.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.navy),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white)
        ]))
        elements.append(t_imp)
        
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Subí tu resumen Credicoop (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analizando geometría del PDF..."):
        df, s_ant, s_fin, res_oficial = procesar_pdf_espacial(uploaded_file.read())
    
    if df.empty:
        st.error("No se encontraron movimientos. Verificá que el PDF sea texto seleccionable.")
    else:
        # Cálculos de Conciliación
        total_cred = df["Crédito"].sum()
        total_deb = df["Débito"].sum()
        saldo_calc = s_ant + total_cred - total_deb
        diff = saldo_calc - s_fin
        
        # 1. VISUALIZACIÓN DE MÉTRICAS
        st.markdown("### 1. Conciliación Bancaria")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Saldo Anterior", fmt_ar(s_ant))
        col2.metric("Créditos (+)", fmt_ar(total_cred))
        col3.metric("Débitos (-)", fmt_ar(total_deb))
        col4.metric("Saldo Calculado", fmt_ar(saldo_calc))
        col5.metric("Saldo PDF", fmt_ar(s_fin), delta=fmt_ar(diff), delta_color="inverse")
        
        if abs(diff) < 1.0:
            st.success("✅ CONCILIACIÓN EXITOSA")
        else:
            st.error(f"❌ DIFERENCIA: {fmt_ar(diff)}. Revisar movimientos no capturados o saldos iniciales.")
        
        # 2. TABLAS Y DATOS
        tab1, tab2, tab3 = st.tabs(["📋 Movimientos Completos", "💰 Gastos e Impuestos", "🏦 Préstamos"])
        
        with tab1:
            st.dataframe(df, use_container_width=True, height=400)
            
        with tab2:
            c_a, c_b = st.columns(2)
            with c_a:
                st.subheader("Según Movimientos (Calculado)")
                gastos_calc = df.groupby("Clasificación")[["Débito"]].sum().reset_index()
                gastos_calc = gastos_calc[gastos_calc["Débito"] > 0]
                gastos_calc["Débito"] = gastos_calc["Débito"].apply(fmt_ar)
                st.dataframe(gastos_calc, use_container_width=True)
            
            with c_b:
                st.subheader("Según Cuadro Oficial (Pie de Página)")
                if res_oficial:
                    df_oficial = pd.DataFrame(res_oficial)
                    df_oficial["Importe"] = df_oficial["Importe"].apply(fmt_ar)
                    st.dataframe(df_oficial, use_container_width=True)
                else:
                    st.warning("No se detectó el cuadro resumen 'Percepciones' al final del PDF.")

        with tab3:
            df_prestamos = df[df["Clasificación"] == "Préstamos"]
            if not df_prestamos.empty:
                st.dataframe(df_prestamos)
                st.info(f"Total pagado en préstamos: {fmt_ar(df_prestamos['Débito'].sum())}")
            else:
                st.info("No se detectaron movimientos de préstamos.")

        # 3. ZONA DE DESCARGAS
        st.markdown("---")
        st.subheader("📥 Exportar Datos")
        
        col_d1, col_d2 = st.columns(2)
        
        # Excel
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Movimientos', index=False)
            if res_oficial:
                pd.DataFrame(res_oficial).to_excel(writer, sheet_name='Resumen_Fiscal', index=False)
        
        col_d1.download_button(
            label="Descargar Excel Completo",
            data=buffer_excel.getvalue(),
            file_name="conciliacion_credicoop.xlsx",
            mime="application/vnd.ms-excel"
        )
        
        # PDF
        metricas_dict = {
            's_ant': fmt_ar(s_ant), 't_cred': fmt_ar(total_cred), 't_deb': fmt_ar(total_deb),
            's_calc': fmt_ar(saldo_calc), 's_fin': fmt_ar(s_fin), 'diff': fmt_ar(diff)
        }
        pdf_report = generar_pdf_reporte(df, res_oficial, metricas_dict)
        col_d2.download_button(
            label="Descargar Reporte PDF",
            data=pdf_report,
            file_name="reporte_conciliacion.pdf",
            mime="application/pdf"
        )
