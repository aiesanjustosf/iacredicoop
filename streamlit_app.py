import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# Configuración inicial para evitar errores si falta fpdf
try:
    from fpdf import FPDF
    TIENE_FPDF = True
except ImportError:
    TIENE_FPDF = False

st.set_page_config(page_title="IA Resumen Bancario", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE LIMPIEZA ---
def limpiar_numero_ar(valor):
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

# --- GENERACIÓN DE PDF (RESUMEN OPERATIVO) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Resumen Operativo - Impuestos y Gastos', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

def generar_pdf_resumen(texto):
    if not TIENE_FPDF: return None
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for line in texto.split('\n'):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_line)
    
    buffer = BytesIO()
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
    except:
        pdf_bytes = pdf.output(dest='S')
    buffer.write(pdf_bytes)
    return buffer.getvalue()

# --- PROCESAMIENTO PRINCIPAL ---
def procesar_pdf(pdf_file, x_coords):
    movimientos = []
    texto_resumen = ""
    saldo_anterior = 0.0
    
    # Desempaquetar sliders
    x_fecha, x_desc, x_debito, x_credito = x_coords

    with pdfplumber.open(pdf_file) as pdf:
        # 1. Buscar Saldo Anterior (Pág 1)
        if len(pdf.pages) > 0:
            p1 = pdf.pages[0].extract_text() or ""
            m = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", p1, re.IGNORECASE)
            if m: saldo_anterior = limpiar_numero_ar(m.group(1))

        # 2. Leer páginas
        fin_grilla = False
        
        for page in pdf.pages:
            texto_pag = page.extract_text() or ""
            
            # Detectar fin de la tabla (Sección Impuestos)
            if "SALDO AL" in texto_pag:
                fin_grilla = True
                parts = texto_pag.split("SALDO AL")
                if len(parts) > 1:
                    texto_resumen += "SALDO AL" + parts[1] + "\n"
            elif fin_grilla:
                texto_resumen += texto_pag + "\n"
            
            # Leer Grilla (mientras no hayamos terminado o estemos en la pagina de transición)
            if not fin_grilla or "SALDO AL" in texto_pag:
                # Definir líneas verticales manuales
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
                        # Validar si es movimiento (Fecha DD/MM/YY)
                        if len(row) >= 5 and re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                            # Evitar leer la línea de totales como movimiento
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

    return pd.DataFrame(movimientos), saldo_anterior, texto_resumen

# --- INTERFAZ DE USUARIO ---

# Encabezado
c1, c2 = st.columns([1, 5])
with c1:
    try:
        st.image("logo_aie.png", width=100)
    except:
        st.warning("Falta logo")
with c2:
    st.title("IA Resumen Bancario – Banco Credicoop")

st.markdown("---")

col_izq, col_der = st.columns([1, 3])

with col_iz
