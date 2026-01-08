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

# --- CONFIGURACIÓN UI (Debe ir primero) ---
st.set_page_config(page_title="IA Resumen Credicoop", layout="wide")

# --- DEBUG: Verificar librerías ---
try:
    import xlsxwriter
except ImportError:
    st.error("❌ Faltan librerías. Agregá 'xlsxwriter' a tu requirements.txt")
    st.stop()

st.title("🏦 IA Resumen Credicoop: Conciliación Estricta")

# --- UTILIDADES ---
def fmt_ar(n):
    if n is None or np.isnan(n): return "—"
    return f"$ {n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def parse_money(text):
    if not text: return 0.0
    text = str(text).replace(" ", "").replace("−", "-").replace("–", "-")
    clean = text.replace(".", "").replace(",", ".")
    try:
        return float(clean)
    except ValueError:
        return 0.0

def clasificar_movimiento(desc):
    u = str(desc).upper()
    if "25413" in u or "25.413" in u: return "Ley 25.413 (Imp. Cheque)"
    if "SIRCREB" in u: return "SIRCREB"
    if "PERCEP" in u and "IVA" in u: return "Percepciones IVA"
    if "I.V.A." in u or "IVA " in u:
        if "10,5" in u: return "IVA 10,5%"
        return "IVA 21%"
    if any(x in u for x in ["COMISION", "MANTEN", "SERVICIO", "GASTOS"]): return "Gastos Bancarios (Neto)"
    if any(x in u for x in ["PRESTAMO", "CUOTA", "AMORTIZACION"]): return "Préstamos"
    return "Otros Movimientos"

# --- LÓGICA CORE ---
def obtener_limites_columnas(pdf):
    # Valores por defecto para A4 Credicoop
    limites = {"debito_start": 350, "debito_end": 460, "credito_end": 530}
    try:
        first_page = pdf.pages[0]
        words = first_page.extract_words()
        deb_box = next((w for w in words if "DEBITO" in w['text'].upper()), None)
        cred_box = next((w for w in words if "CREDITO" in w['text'].upper()), None)
        
        if deb_box and cred_box:
            limites["debito_start"] = deb_box['x0'] - 20
            limites["debito_end"] = (deb_box['x1'] + cred_box['x0']) / 2
            limites["credito_end"] = cred_box['x1'] + 40
    except Exception as e:
        print(f"Advertencia: No se pudieron detectar columnas dinámicas ({e}). Usando defaults.")
    return limites

def procesar_pdf_espacial(pdf_bytes):
    rows = []
    saldo_inicial = 0.0
    saldo_final = 0.0
    resumen_impositivo_oficial = []
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        limites = obtener_limites_columnas(pdf)
        full_text_for_footer = ""
        
        for page in pdf.pages:
            full_text_for_footer += (page.extract_text() or "") + "\n"
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            lines = {}
            for w in words:
                top = round(w['top'])
                if top not in lines: lines[top] = []
                lines[top].append(w)
            
            sorted_y = sorted(lines.keys())
            
            for y in sorted_y:
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words])
                
                # Regex para detectar fecha al inicio (dd/mm/yy o dd/mm/yyyy)
                match_date = re.match(r"^(\d{2}/\d{2}/\d{2,4})", line_text)
                
                if match_date:
                    fecha = match_date.group(1)
                    debito = 0.0
                    credito = 0.0
                    descripcion_parts = []
                    
                    for w in line_words:
                        text = w['text']
                        x_center = (w['x0'] + w['x1']) / 2
                        
                        # Si es dinero
                        if re.match(r"^-?[\d\.]+,[\d]{2}$", text):
                            valor = parse_money(text)
                            if limites["debito_start"] < x_center < limites["debito_end"]:
                                debito = valor
                            elif limites["debito_end"] < x_center < limites["credito_end"]:
                                credito = valor
                        else:
                            # Si no es fecha ni parte del comprobante (simple heurística)
                            if text != fecha and not (text.isdigit() and len(text) > 4 and line_words.index(w) < 2):
                                descripcion_parts.append(text)
                    
                    desc_final = " ".join(descripcion_parts)
                    rows.append({
                        "Fecha": fecha, "Descripción": desc_final, 
                        "Débito": debito, "Crédito": credito, 
                        "Clasificación": clasificar_movimiento(desc_final)
                    })
                
                # Detectar Saldo Anterior
                elif "SALDO" in line_text and "ANTERIOR" in line_text:
                    try:
                        nums = [w['text'] for w in line_words if re.match(r"-?[\d\.]+,[\d]{2}", w['text'])]
                        if nums: saldo_inicial = parse_money(nums[-1])
                    except: pass

        # Buscar Saldo Final
        match_saldo_fin = re.search(r"SALDO AL \d{2}/\d{2}/\d{2,4}\s+([\d\.,\-]+)", full_text_for_footer)
        if match_saldo_fin:
            saldo_final = parse_money(match_saldo_fin.group(1))
            
        # Cuadros de Impuestos
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
                if monto > 0: resumen_impositivo_oficial.append({"Concepto": name, "Importe": monto})

    return pd.DataFrame(rows), saldo_inicial, saldo_final, resumen_impositivo_oficial

def generar_pdf_reporte(df, resumen_imp, metricas):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Informe de Conciliación Bancaria - Credicoop", styles['Title']))
    elements.append(Spacer(1, 12))
    
    data_metrics = [
        ["Saldo Anterior", metricas['s_ant']],
        ["Total Créditos", metricas['t_cred']],
        ["Total Débitos", metricas['t_deb']],
        ["Saldo Calculado", metricas['s_calc']],
        ["Saldo Resumen", metricas['s_fin']],
        ["Diferencia", metricas['diff']]
    ]
    t_met = Table(data_metrics, colWidths=[200, 150])
    t_met.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)]))
    elements.append(t_met)
    
    if resumen_imp:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Cuadro de Impuestos Oficial", styles['Heading2']))
        data_imp = [["Concepto", "Importe"]] + [[r['Concepto'], fmt_ar(r['Importe'])] for r in resumen_imp]
        t_imp = Table(data_imp, colWidths=[300, 100])
        t_imp.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.navy), ('TEXTCOLOR', (0,0), (-1,0), colors.white)]))
        elements.append(t_imp)
        
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- APP PRINCIPAL ---
uploaded_file = st.file_uploader("Subí tu resumen Credicoop (PDF)", type="pdf")

if uploaded_file:
    # BLOQUE DE SEGURIDAD (TRY/EXCEPT)
    try:
        with st.spinner("Procesando PDF (detectando columnas y movimientos)..."):
            df, s_ant, s_fin, res_oficial = procesar_pdf_espacial(uploaded_file.read())
        
        if df.empty:
            st.warning("⚠️ El PDF fue procesado pero no se encontraron movimientos. ¿Es un PDF escaneado (imagen)? Esta app requiere texto seleccionable.")
        else:
            # Cálculos
            total_cred = df["Crédito"].sum()
            total_deb = df["Débito"].sum()
            saldo_calc = s_ant + total_cred - total_deb
            diff = saldo_calc - s_fin
            
            # Métricas
            st.markdown("### 1. Conciliación Bancaria")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Saldo Anterior", fmt_ar(s_ant))
            c2.metric("Créditos", fmt_ar(total_cred))
            c3.metric("Débitos", fmt_ar(total_deb))
            c4.metric("Diferencia", fmt_ar(diff), delta_color="inverse")

            if abs(diff) < 1.0:
                st.success(f"✅ Conciliación OK (Saldo Calc: {fmt_ar(saldo_calc)} vs PDF: {fmt_ar(s_fin)})")
            else:
                st.error(f"❌ Diferencia de {fmt_ar(diff)}")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["📋 Grilla Completa", "💰 Gastos e Impuestos", "🏦 Préstamos"])
            
            with tab1:
                st.dataframe(df, use_container_width=True, height=500)
                
            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Cálculo según clasificación de movimientos")
                    gastos = df.groupby("Clasificación")[["Débito"]].sum().reset_index()
                    st.dataframe(gastos[gastos["Débito"] > 0].style.format({"Débito": fmt_ar}), use_container_width=True)
                with col_b:
                    st.caption("Extracción del cuadro oficial (pie de página)")
                    if res_oficial:
                        st.dataframe(pd.DataFrame(res_oficial).style.format({"Importe": fmt_ar}), use_container_width=True)
                    else:
                        st.info("No se encontró el cuadro resumen al final del PDF.")

            with tab3:
                prest = df[df["Clasificación"] == "Préstamos"]
                if not prest.empty:
                    st.dataframe(prest)
                else:
                    st.info("Sin préstamos.")

            # Descargas
            st.divider()
            d1, d2 = st.columns(2)
            
            # Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Movimientos', index=False)
                if res_oficial: pd.DataFrame(res_oficial).to_excel(writer, sheet_name='Fiscal', index=False)
            
            d1.download_button("Descargar Excel", data=excel_buffer.getvalue(), file_name="conciliacion.xlsx", mime="application/vnd.ms-excel")
            
            # PDF
            mets = {'s_ant': fmt_ar(s_ant), 't_cred': fmt_ar(total_cred), 't_deb': fmt_ar(total_deb), 's_calc': fmt_ar(saldo_calc), 's_fin': fmt_ar(s_fin), 'diff': fmt_ar(diff)}
            pdf_data = generar_pdf_reporte(df, res_oficial, mets)
            d2.download_button("Descargar Reporte PDF", data=pdf_data, file_name="reporte.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"❌ Ocurrió un error crítico al procesar el archivo:")
        st.exception(e)
