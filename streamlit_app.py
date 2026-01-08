import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
from io import BytesIO
from fpdf import FPDF

# --- CONFIGURACIÓN DE PÁGINA Y ESTILO ---
st.set_page_config(page_title="IA Resumen Bancario - AIE", layout="wide")

# CSS para tarjetas y estilo
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---

def limpiar_numero_ar(valor):
    """Convierte '1.000,00' a float."""
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

# --- GENERADOR DE PDF (RESUMEN IMPOSITIVO) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Resumen Operativo - Impuestos y Tasas', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

def generar_pdf_resumen(texto):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for line in texto.split('\n'):
        if line.strip():
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, safe_line)
    
    buffer = BytesIO()
    # Output devuelve string en versiones viejas o bytes en nuevas, ajustamos:
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
    except:
        pdf_bytes = pdf.output(dest='S')
        
    buffer.write(pdf_bytes)
    return buffer.getvalue()

# --- PROCESAMIENTO PRINCIPAL ---

def procesar_pdf(pdf_file, x_coords):
    movimientos = []
    texto_impuestos = ""
    saldo_anterior = 0.0
    
    # Coordenadas manuales (Sliders)
    x_fecha, x_desc, x_debito, x_credito = x_coords
    
    with pdfplumber.open(pdf_file) as pdf:
        # 1. Saldo Anterior (Pág 1)
        if len(pdf.pages) > 0:
            p1 = pdf.pages[0].extract_text() or ""
            m = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", p1, re.IGNORECASE)
            if m: saldo_anterior = limpiar_numero_ar(m.group(1))

        # 2. Procesar páginas
        fin_grilla = False
        
        for page in pdf.pages:
            p_text = page.extract_text() or ""
            
            # Detectar corte "SALDO AL"
            if "SALDO AL" in p_text and not fin_grilla:
                fin_grilla = True
                parts = p_text.split("SALDO AL")
                if len(parts) > 1:
                    texto_impuestos += "SALDO AL" + parts[1] + "\n"
            elif fin_grilla:
                texto_impuestos += p_text + "\n"
            
            # Extraer tabla (Solo si no terminó la grilla o es la página de transición)
            if not fin_grilla or "SALDO AL" in p_text:
                # Líneas verticales explícitas
                lines = [0, x_fecha, x_desc, x_debito, x_credito, page.width]
                settings = {
                    "vertical_strategy": "explicit", 
                    "explicit_vertical_lines": lines,
                    "horizontal_strategy": "text",
                    "intersection_y_tolerance": 5
                }
                
                table = page.extract_table(settings)
                
                if table:
                    for row in table:
                        row = [c.strip() if c else "" for c in row]
                        # Validar fecha al inicio y evitar leer la linea de "SALDO AL" dentro de la tabla
                        if len(row) >= 5 and re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                            if "SALDO AL" in row[1]: continue
                            try:
                                movimientos.append({
                                    "Fecha": row[0],
                                    "Descripcion": row[1],
                                    "Debito": limpiar_numero_ar(row[2]),
                                    "Credito": limpiar_numero_ar(row[3]),
                                    "Saldo_PDF": limpiar_numero_ar(row[4])
                                })
                            except: pass

    return pd.DataFrame(movimientos), saldo_anterior, texto_impuestos

# --- INTERFAZ GRÁFICA ---

# 1. Cabecera con Logo
col_l, col_t = st.columns([1, 5])
with col_l:
    if os.path.exists("logo_aie.png"):
        st.image("logo_aie.png", width=120)
    else:
        st.warning("⚠️ Subí 'logo_aie.png'")
with col_t:
    st.title("IA Resumen Bancario – Banco Credicoop")

st.markdown("---")

col_conf, col_main = st.columns([1, 3])

with col_conf:
    st.header("⚙️ Ajustes")
    st.info("Si la tabla sale vacía o cortada, mové estos controles:")
    
    # SLIDERS (Fundamental para que ande siempre)
    x_fecha = st.slider("Fin Fecha", 0, 120, 60)
    x_desc = st.slider("Fin Descripción", 100, 500, 310, help="Achicá este número si te come los Créditos")
    x_debito = st.slider("Fin Débito", 300, 600, 480)
    x_credito = st.slider("Fin Crédito", 400, 700, 580)
    
    uploaded_file = st.file_uploader("Subir PDF", type="pdf")

if uploaded_file:
    # Procesar
    x_coords = [x_fecha, x_desc, x_debito, x_credito]
    df, saldo_ini_auto, txt_resumen = procesar_pdf(uploaded_file, x_coords)
    
    with col_conf:
        saldo_inicial = st.number_input("Saldo Anterior", value=saldo_ini_auto, step=1000.0)

    # Validar si hay datos
    if df.empty:
        st.warning("⚠️ No se leyeron datos. Por favor, ajustá los Sliders a la izquierda.")
    else:
        # Conciliar
        df['Saldo_Calculado'] = 0.0
        df['Estado'] = 'OK'
        df['Diferencia'] = 0.0
        
        acum = saldo_inicial
        t_cred = df['Credito'].sum()
        t_deb = df['Debito'].sum()
        
        for i, row in df.iterrows():
            acum += (row['Credito'] - row['Debito'])
            df.at[i, 'Saldo_Calculado'] = acum
            
            if row['Saldo_PDF'] != 0:
                diff = round(acum - row['Saldo_PDF'], 2)
                if abs(diff) > 1.00:
                    df.at[i, 'Estado'] = 'ERROR'
                    df.at[i, 'Diferencia'] = diff
                else:
                    acum = row['Saldo_PDF']
        
        saldo_final = acum

        # --- RESULTADOS ---
        with col_main:
            # Tarjetas Métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Saldo Anterior", formatear_moneda_ar(saldo_inicial))
            m2.metric("Créditos", formatear_moneda_ar(t_cred))
            m3.metric("Débitos", formatear_moneda_ar(t_deb))
            
            errores = df[df['Estado'] == 'ERROR']
            if not errores.empty:
                m4.metric("Saldo Final", formatear_moneda_ar(saldo_final), "Diferencia", delta_color="inverse")
                st.error(f"❌ {len(errores)} Errores de Conciliación")
            else:
                m4.metric("Saldo Final", formatear_moneda_ar(saldo_final), "Ok")

            # PESTAÑAS: MOVIMIENTOS Y RESUMEN
            tab_mov, tab_res = st.tabs(["📂 Movimientos (Excel)", "📑 Resumen Impositivo (PDF)"])
            
            with tab_mov:
                if not errores.empty:
                    st.dataframe(errores[['Fecha','Descripcion','Saldo_PDF','Saldo_Calculado','Diferencia']])
                
                df_show = df.copy()
                for c in ['Debito','Credito','Saldo_PDF','Saldo_Calculado','Diferencia']:
                    df_show[c] = df_show[c].apply(formatear_moneda_ar)
                
                st.dataframe(df_show, use_container_width=True, height=450)
                
                # Botón Excel
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                st.download_button("📥 Descargar Excel", buffer.getvalue(), "conciliacion.xlsx")

            with tab_res:
                st.subheader("Impuestos, Tasas y Comisiones")
                if txt_resumen.strip():
                    col_txt, col_dl = st.columns([3, 1])
                    with col_txt:
                        st.text_area("Texto Detectado", txt_resumen, height=300)
                    with col_dl:
                        st.write("Generar PDF:")
                        pdf_data = generar_pdf_resumen(txt_resumen)
                        st.download_button("📄 Descargar Resumen", pdf_data, "resumen_impositivo.pdf", mime="application/pdf", type="primary")
                else:
                    st.info("No se encontró la sección 'SALDO AL' para extraer impuestos.")

else:
    with col_main:
        st.info("Esperando archivo...")
