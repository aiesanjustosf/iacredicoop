import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="IA Resumen Bancario - Credicoop", layout="wide")

# --- FUNCIONES DE LIMPIEZA Y FORMATO ---

def limpiar_numero_ar(valor):
    """Convierte string formato '1.000,00' a float de forma segura."""
    if not valor: return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    
    val_str = str(valor).strip()
    # Manejo de negativos y limpieza
    es_negativo = False
    if val_str.endswith("-") or (val_str.startswith("(") and val_str.endswith(")")):
        es_negativo = True
    
    val_str = re.sub(r'[^\d,.]', '', val_str)
    if not val_str: return 0.0
        
    try:
        # Formato AR: 1.000,00 -> 1000.00
        val_str = val_str.replace(".", "").replace(",", ".")
        num = float(val_str)
        return -num if es_negativo else num
    except:
        return 0.0

def formatear_moneda_ar(valor):
    """Visualización estilo AR: 1.000,00"""
    if pd.isna(valor) or valor == "": return ""
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- PROCESAMIENTO DEL PDF (CON SLIDERS) ---

def procesar_pdf(pdf_file, x_coords):
    """
    x_coords: [x_fecha, x_desc, x_debito, x_credito]
    """
    data = []
    saldo_anterior = 0.0
    columnas_base = ["Fecha", "Descripcion", "Debito", "Credito", "Saldo_PDF"]
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            # 1. Intentar leer Saldo Anterior de la pág 1
            if len(pdf.pages) > 0:
                text_p1 = pdf.pages[0].extract_text() or ""
                match_saldo = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", text_p1, re.IGNORECASE)
                if match_saldo:
                    saldo_anterior = limpiar_numero_ar(match_saldo.group(1))

            # 2. Leer tablas usando las líneas verticales manuales
            for page in pdf.pages:
                # Definimos las líneas de corte explícitas
                vertical_lines = [0] + x_coords + [page.width]
                
                table_settings = {
                    "vertical_strategy": "explicit",
                    "explicit_vertical_lines": vertical_lines,
                    "horizontal_strategy": "text",
                    "intersection_y_tolerance": 5, 
                }
                
                table = page.extract_table(table_settings)
                
                if table:
                    for row in table:
                        row = [c.strip() if c else "" for c in row]
                        
                        # Validar que sea una línea de movimiento (empieza con fecha)
                        if len(row) >= 5 and re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                            try:
                                data.append({
                                    "Fecha": row[0],
                                    "Descripcion": row[1],
                                    "Debito": limpiar_numero_ar(row[2]),
                                    "Credito": limpiar_numero_ar(row[3]),
                                    "Saldo_PDF": limpiar_numero_ar(row[4])
                                })
                            except:
                                pass # Ignorar filas con basura

    except Exception as e:
        return pd.DataFrame(columns=columnas_base), 0.0, str(e)

    if not data:
        return pd.DataFrame(columns=columnas_base), saldo_anterior, "No data"
        
    return pd.DataFrame(data), saldo_anterior, "OK"

# --- LÓGICA DE CONCILIACIÓN ---

def verificar_conciliacion(df, saldo_ini):
    if df.empty: return df, 0, 0, 0
    
    df['Saldo_Calculado'] = 0.0
    df['Estado'] = 'OK'
    df['Diferencia'] = 0.0
    
    acumulado = saldo_ini
    t_cred = df['Credito'].sum()
    t_deb = df['Debito'].sum()
    
    for i, row in df.iterrows():
        acumulado += (row['Credito'] - row['Debito'])
        df.at[i, 'Saldo_Calculado'] = acumulado
        
        # Checkpoint: Comparar con la columna Saldo del PDF
        saldo_pdf = row['Saldo_PDF']
        if saldo_pdf != 0:
            diff = round(acumulado - saldo_pdf, 2)
            if abs(diff) > 1.00:
                df.at[i, 'Estado'] = 'ERROR'
                df.at[i, 'Diferencia'] = diff
            else:
                acumulado = saldo_pdf # Sincronizar para evitar arrastre
                
    return df, t_cred, t_deb, acumulado

# --- INTERFAZ (UI) ---

# Intento de cargar logo si existe, sino sigue de largo
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    try:
        st.image("logo_aie.png", width=100) # Si tenés el archivo, ponelo en la carpeta
    except:
        st.write("🏦") # Placeholder si no hay logo

with col_titulo:
    st.title("IA Resumen Bancario – Banco Credicoop")

st.write("Subí un PDF del resumen bancario (Banco Credicoop)")

col_config, col_main = st.columns([1, 3])

with col_config:
    st.markdown("### 🛠️ Calibración")
    st.info("Ajustá las columnas aquí si faltan datos.")
    
    # Sliders manuales (Clave para arreglar tu problema)
    x_fecha = st.slider("Fin Fecha", 0, 150, 60)
    x_desc = st.slider("Fin Descripción", 100, 500, 340, help="Si se corta un crédito, mové esto a la izquierda")
    x_debito = st.slider("Fin Débito", 300, 600, 480)
    x_credito = st.slider("Fin Crédito", 400, 700, 580)
    
    uploaded_file = st.file_uploader("Cargar PDF", type="pdf")

if uploaded_file:
    # 1. Procesar
    x_coords = [x_fecha, x_desc, x_debito
