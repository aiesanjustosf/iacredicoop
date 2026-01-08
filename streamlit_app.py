import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# --- CONFIGURACIÓN VISUAL Y DE PÁGINA ---
st.set_page_config(page_title="Conciliador Credicoop", layout="wide")

# CSS para recuperar la estética de tarjetas limpias
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE FORMATO ---

def limpiar_numero_ar(valor):
    """Convierte '1.000,00' a float de forma segura."""
    if not valor: return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    
    val_str = str(valor).strip()
    es_negativo = False
    
    # Detectar negativos (formato 1.000,00- o (1.000,00))
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
    """Visualización 1.000,00"""
    if pd.isna(valor) or valor == "": return ""
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- LÓGICA INTELIGENTE ---

def detectar_columnas_auto(pdf_page):
    """Busca las palabras clave en la cabecera para definir las columnas x automaticamente"""
    words = pdf_page.extract_words()
    
    # Valores por defecto (Fallback) si no encuentra las palabras
    x_desc = 320
    x_debito = 480
    x_credito = 580
    
    found_deb = False
    found_cred = False
    found_saldo = False

    # Buscamos las cabeceras
    for w in words:
        text = w['text'].lower()
        if "descripción" in text or "concepto" in text:
            # El fin de la descripción suele ser un poco antes de donde empieza el débito
            x_desc = w['x1'] + 50 
        if "débito" in text or "debito" in text:
            x_debito = w['x0'] - 10 # Un poco a la izquierda de la palabra
            found_deb = True
        if "crédito" in text or "credito" in text:
            x_credito = w['x0'] - 10
            found_cred = True
            
    # Lógica de corrección: Si encontró Débito, usalo para ajustar Descripción
    if found_deb:
        x_desc = x_debito - 10

    return [x_desc, x_debito, x_credito]

def procesar_pdf(pdf_file, use_auto, manual_coords=None):
    data = []
    saldo_anterior = 0.0
    columns = ["Fecha", "Descripcion", "Debito", "Credito", "Saldo_PDF"] # Estructura base
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            if len(pdf.pages) == 0: return pd.DataFrame(columns=columns), 0.0
            
            # 1. Obtener Saldo Anterior (Regex en pág 1)
            text_p1 = pdf.pages[0].extract_text() or ""
            match_saldo = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", text_p1, re.IGNORECASE)
            if match_saldo:
                saldo_anterior = limpiar_numero_ar(match_saldo.group(1))

            # 2. Definir coordenadas de corte
            if use_auto:
                # Detectar en la primera página
                coords = detectar_columnas_auto(pdf.pages[0])
            else:
                coords = manual_coords

            # Líneas verticales: [Inicio, FinDesc/InicioDeb, FinDeb/InicioCred, FinCred/InicioSaldo]
            # Ajustamos para extract_table: [0, x_desc, x_debito, x_credito, page_width]
            
            for page in pdf.pages:
                vertical_lines = [0] + coords + [page.width]
                
                table_settings = {
                    "vertical_strategy": "explicit",
                    "explicit_vertical_lines": vertical_lines,
                    "horizontal_strategy": "text",
                    "intersection_y_tolerance": 5, 
                }
                
                table = page.extract_table(table_settings)
                
                if table:
                    for row in table:
                        # Limpieza y validación
                        row = [c.strip() if c else "" for c in row]
                        
                        # Debe tener fecha válida (DD/MM/YY) en la primera columna
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
                                continue

    except Exception as e:
        st.error(f"Error leyendo el PDF: {e}")
        return pd.DataFrame(columns=columns), 0.0

    if not data:
        return pd.DataFrame(columns=columns), saldo_anterior
        
    return pd.DataFrame(data), saldo_anterior

def calcular_conciliacion(df, saldo_ini):
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
        
        # Checkpoint con columna Saldo del PDF
        saldo_pdf = row['Saldo_PDF']
        if saldo_pdf != 0:
            diff = round(acumulado - saldo_pdf, 2)
            if abs(diff) > 1.00:
                df.at[i, 'Estado'] = 'ERROR'
                df.at[i, 'Diferencia'] = diff
            else:
                acumulado = saldo_pdf # Sincronizar
                
    return df, t_cred, t_deb, acumulado

# --- INTERFAZ DE USUARIO ---

st.title("🏦 Conciliador Automático Credicoop")

# Panel lateral o Expander para "Ajustes Finos" (oculto por defecto)
with st.expander("⚙️ Configuración Avanzada (Si no lee bien las columnas)", expanded=False):
    col_opt, col_sliders = st.columns([1, 3])
    with col_opt:
        modo_auto = st.checkbox("Detección Automática", value=True, help="Intenta buscar los encabezados Débito/Crédito solo")
    
    with col_sliders:
        st.caption("Ajustar líneas verticales manualmente:")
        x_desc = st.slider("Fin Descripción / Inicio Débito", 200, 500, 320)
        x_deb = st.slider("Fin Débito / Inicio Crédito", 400, 600, 480)
        x_cred = st.slider("Fin Crédito / Inicio Saldo", 500, 700, 580)

# Carga de Archivo
uploaded_file = st.file_uploader("Arrastrá y soltá el PDF aquí", type="pdf")

if uploaded_file:
    # Procesar
    manual_coords = [x_desc, x_deb, x_cred]
    df_raw, saldo_detectado = procesar_pdf(uploaded_file, modo_auto, manual_coords)
    
    # Input Saldo Anterior (Editable)
    col_saldo, _ = st.columns([1, 3])
    with col_saldo:
        saldo_inicial = st.number_input("Saldo Anterior", value=saldo_detectado, step=1000.0)

    # Validar si hay datos
    if df_raw.empty:
        st.warning("⚠️ No se pudieron leer movimientos.")
        st.info("Intento de solución: Abre la 'Configuración Avanzada' arriba, desactiva 'Detección Automática' y mueve los sliders hasta que coincidan con las columnas de tu PDF.")
    else:
        # Calcular
        df_final, tot_cred, tot_deb, saldo_final = calcular_conciliacion(df_raw, saldo_inicial)
        
        st.divider()

        # TARJETAS DE MÉTRICAS (Diseño visual)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Saldo Anterior", formatear_moneda_ar(saldo_inicial))
        c2.metric("Total Créditos", formatear_moneda_ar(tot_cred), delta="Ingresos")
        c3.metric("Total Débitos", formatear_moneda_ar(tot_deb), delta="-Egresos", delta_color="inverse")
        
        # Color del Saldo Final según integridad
        errores = df_final[df_final['Estado'] == 'ERROR']
        if not errores.empty:
            c4.metric("Saldo Calculado", formatear_moneda_ar(saldo_final), "Con Diferencias", delta_color="inverse")
            
            # ALERTA VISUAL
            st.error(f"❌ ERROR DE CONCILIACIÓN: Hay {len(errores)} movimientos donde el saldo no coincide.")
            st.write("Diferencias detectadas (Saldo Calculado vs Saldo Real del PDF):")
            
            view_err = errores[['Fecha', 'Descripcion', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']].copy()
            for c in view_err.columns[2:]: view_err[c] = view_err[c].apply(formatear_moneda_ar)
            st.dataframe(view_err, use_container_width=True)
            
        else:
            c4.metric("Saldo Calculado", formatear_moneda_ar(saldo_final), "Conciliado OK")
            st.success("✅ Conciliación Perfecta. Los saldos parciales coinciden con el banco.")

        # TABLA PRINCIPAL
        st.subheader("Detalle de Movimientos")
        
        # Formateo para visualización
        df_display = df_final.copy()
        cols_fmt = ['Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']
        for c in cols_fmt:
            df_display[c] = df_display[c].apply(formatear_moneda_ar)
            
        st.dataframe(df_display, use_container_width=True, height=500)
        
        # EXPORTAR
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, index=False, sheet_name='Conciliacion')
            
        st.download_button(
            "📥 Descargar Excel (.xlsx)", 
            data=buffer.getvalue(), 
            file_name="conciliacion_credicoop.xlsx", 
            mime="application/vnd.ms-excel",
            type="primary"
        )
else:
    # Mensaje de bienvenida limpio
    st.info("👆 Subí el archivo PDF para comenzar el análisis automático.")
