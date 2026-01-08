import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO
from fpdf import FPDF

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Resumen Bancario", layout="wide")

# --- FUNCIONES DE LIMPIEZA Y FORMATO ---

def limpiar_numero_ar(valor):
    """Convierte '1.000,00' a float de forma segura."""
    if not valor: return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    
    val_str = str(valor).strip()
    es_negativo = False
    if val_str.endswith("-") or (val_str.startswith("(") and val_str.endswith(")")):
        es_negativo = True
    
    val_str = re.sub(r'[^\d,.]', '', val_str)
    if not val_str: return 0.0
        
    try:
        val_str = val_str.replace(".", "").replace(",", ".")
        num = float(val_str)
        return -num if es_negativo else num
    except:
        return 0.0

def formatear_moneda_ar(valor):
    if pd.isna(valor) or valor == "": return ""
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- GENERACIÓN DE PDF RESUMEN OPERATIVO ---

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Resumen Operativo - Impuestos y Gastos', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

def generar_pdf_resumen(texto_resumen):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Procesar texto línea por línea
    lines = texto_resumen.split('\n')
    for line in lines:
        if line.strip():
            # Intentar detectar títulos vs datos
            if "IMPUESTO" in line or "IVA" in line or "PERCEPCION" in line:
                 pdf.set_font("Arial", 'B', 10)
            else:
                 pdf.set_font("Arial", '', 10)
            
            # Limpieza de caracteres raros
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 8, safe_line)
            
    buffer = BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin-1')
    buffer.write(pdf_content)
    return buffer.getvalue()

# --- MOTOR DE LECTURA INTELIGENTE ---

def procesar_completo(pdf_file):
    movimientos = []
    texto_resumen_operativo = ""
    saldo_anterior = 0.0
    
    # Banderas de control
    fin_grilla_detectado = False
    
    # COORDENADAS FIJAS OPTIMIZADAS (Credicoop Estándar)
    # Ajusté "Fin Descripción" a 310 para que no se coma los créditos largos
    x_coords = [0, 60, 310, 480, 580, 1000] 
    
    with pdfplumber.open(pdf_file) as pdf:
        # 1. Buscar Saldo Anterior (Página 1)
        if len(pdf.pages) > 0:
            p1_text = pdf.pages[0].extract_text()
            match = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", p1_text, re.IGNORECASE)
            if match:
                saldo_anterior = limpiar_numero_ar(match.group(1))

        # 2. Iterar Páginas
        for page in pdf.pages:
            # A) Si ya terminó la grilla, todo es resumen operativo
            if fin_grilla_detectado:
                texto_resumen_operativo += page.extract_text() + "\n"
                continue
            
            # B) Si estamos en la grilla, buscamos "SALDO AL"
            page_text = page.extract_text()
            if "SALDO AL" in page_text:
                # Aquí pasa la transición
                fin_grilla_detectado = True
                
                # Separar texto: Lo que está después de "SALDO AL" va al resumen
                parts = page_text.split("SALDO AL")
                if len(parts) > 1:
                    # La parte derecha es el inicio del resumen (fecha y saldo final)
                    texto_resumen_operativo += "SALDO AL" + parts[1] + "\n"
                
                # Intentamos leer la tabla HASTA ese punto (o la página entera, filtrando después)
                # Por seguridad, procesamos esta página como tabla también
            
            # C) Extracción de Tabla (Grilla)
            settings = {
                "vertical_strategy": "explicit",
                "explicit_vertical_lines": x_coords,
                "horizontal_strategy": "text",
                "intersection_y_tolerance": 5
            }
            table = page.extract_table(settings)
            
            if table:
                for row in table:
                    # Limpieza básica
                    row = [c.strip() if c else "" for c in row]
                    
                    # FILTRO: Solo es movimiento si empieza con fecha DD/MM/YY
                    if len(row) >= 5 and re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                        try:
                            # Ignorar línea si dice "SALDO AL" dentro de la tabla
                            if "SALDO AL" in row[1]: continue
                            
                            movimientos.append({
                                "Fecha": row[0],
                                "Descripcion": row[1],
                                "Debito": limpiar_numero_ar(row[2]),
                                "Credito": limpiar_numero_ar(row[3]),
                                "Saldo_PDF": limpiar_numero_ar(row[4])
                            })
                        except:
                            pass

    df = pd.DataFrame(movimientos)
    return df, saldo_anterior, texto_resumen_operativo

# --- INTERFAZ DE USUARIO ---

# Cabecera
c_logo, c_tit = st.columns([1, 5])
with c_logo:
    try:
        st.image("logo_aie.png", width=100)
    except:
        st.write("📂")
with c_tit:
    st.title("IA Resumen Bancario – Banco Credicoop")

st.markdown("---")

uploaded_file = st.file_uploader("Subí tu Resumen (PDF)", type="pdf")

if uploaded_file:
    with st.spinner('Procesando grilla y detectando impuestos...'):
        df, saldo_ini, texto_resumen = procesar_completo(uploaded_file)
    
    # Sección Saldo (Editable)
    col_input, col_kpi = st.columns([1, 3])
    with col_input:
        saldo_inicial = st.number_input("Saldo Anterior", value=saldo_ini, step=1000.0)
        
    if df.empty:
        st.error("⚠️ No se pudieron leer movimientos. El PDF podría ser una imagen escaneada.")
    else:
        # --- LÓGICA DE CONCILIACIÓN ---
        df['Saldo_Calculado'] = 0.0
        df['Estado'] = 'OK'
        df['Diferencia'] = 0.0
        
        acum = saldo_inicial
        total_cred = df['Credito'].sum()
        total_deb = df['Debito'].sum()
        
        for i, row in df.iterrows():
            acum += (row['Credito'] - row['Debito'])
            df.at[i, 'Saldo_Calculado'] = acumul
            
            if row['Saldo_PDF'] != 0:
                diff = round(acum - row['Saldo_PDF'], 2)
                if abs(diff) > 1.00:
                    df.at[i, 'Estado'] = 'ERROR'
                    df.at[i, 'Diferencia'] = diff
                else:
                    acum = row['Saldo_PDF'] # Sincronizar
        
        saldo_final = acumul

        # --- MOSTRAR RESULTADOS ---
        with col_kpi:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Saldo Ant.", formatear_moneda_ar(saldo_inicial))
            m2.metric("Créditos", formatear_moneda_ar(total_cred))
            m3.metric("Débitos", formatear_moneda_ar(total_deb))
            
            # Alerta de diferencia
            errores = df[df['Estado'] == 'ERROR']
            if not errores.empty:
                m4.metric("Saldo Final", formatear_moneda_ar(saldo_final), "Diferencia", delta_color="inverse")
                st.toast(f"⚠️ Hay {len(errores)} errores de conciliación", icon="❌")
            else:
                m4.metric("Saldo Final", formatear_moneda_ar(saldo_final), "Ok")

        # --- TABS: GRILLA Y RESUMEN ---
        tab1, tab2 = st.tabs(["📊 Movimientos (Excel)", "📑 Resumen Operativo (Impuestos)"])
        
        with tab1:
            st.subheader("Conciliación de Movimientos")
            if not errores.empty:
                st.error("Filas con diferencias de saldo:")
                st.dataframe(errores[['Fecha', 'Descripcion', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']])
            
            # Tabla completa
            df_show = df.copy()
            for c in ['Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']:
                df_show[c] = df_show[c].apply(formatear_moneda_ar)
            st.dataframe(df_show, height=400, use_container_width=True)
            
            # Descarga Excel
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button(
                "📥 Descargar Excel (.xlsx)", 
                data=buffer_excel.getvalue(), 
                file_name="conciliacion_credicoop.xlsx",
                mime="application/vnd.ms-excel"
            )

        with tab2:
            st.subheader("Resumen de Impuestos, Tasas y Comisiones")
            st.info("Este texto se extrajo del final del resumen (después de 'SALDO AL').")
            
            col_txt, col_pdf = st.columns([3, 1])
            
            with col_txt:
                st.text_area("Vista Previa Texto", texto_resumen, height=300)
            
            with col_pdf:
                st.write("Generar reporte PDF:")
                # Generar PDF en memoria
                if texto_resumen.strip():
                    pdf_bytes = generar_pdf_resumen(texto_resumen)
                    st.download_button(
                        "📄 Descargar Resumen (.pdf)",
                        data=pdf_bytes,
                        file_name="resumen_operativo_impositivo.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                else:
                    st.warning("No se detectó resumen operativo en este archivo.")

else:
    st.info("Esperando archivo PDF...")
