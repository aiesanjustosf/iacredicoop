import streamlit as st
import pandas as pd
import pdfplumber
import re
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Procesador Credicoop IA",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #005f9e;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---

def detectar_x_corte(page):
    """
    Busca la palabra 'CREDITO', 'HABER' o 'DEPÓSITOS' en la cabecera
    para determinar automáticamente dónde empieza la columna derecha.
    """
    words = page.extract_words()
    # Buscamos palabras clave de cabecera
    keywords = ["CREDITO", "CRÉDITO", "HABER", "DEPOSITOS"]
    
    for w in words:
        # Si encontramos el encabezado de la columna Credito
        if w['text'].upper() in keywords:
            # Retornamos su posición izquierda (x0) menos un pequeño margen
            return w['x0'] - 10
            
    return None # No se encontró referencia

def procesar_pdf_inteligente(pdf_file, x_corte_manual=None, usar_auto=True):
    datos = []
    # Regex para números argentinos: 1.000,00 o -50,00
    patron_numero = re.compile(r'^-?[\d\.]+,\d{2}$') 

    with pdfplumber.open(pdf_file) as pdf:
        
        # Intentamos detectar el corte automáticamente en la primera página
        x_corte_auto = None
        if usar_auto:
            for page in pdf.pages:
                detectado = detectar_x_corte(page)
                if detectado:
                    x_corte_auto = detectado
                    break # Encontramos la referencia
        
        # Decisión final de qué corte usar
        if usar_auto and x_corte_auto:
            x_corte_final = x_corte_auto
            st.toast(f"🤖 Calibración automática detectada en X: {int(x_corte_final)}")
        else:
            x_corte_final = x_corte_manual if x_corte_manual else 420

        for page in pdf.pages:
            # Extraer palabras
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            # Agrupar por renglón
            filas = {}
            for w in words:
                y_pos = round(w['top']) 
                if y_pos not in filas:
                    filas[y_pos] = []
                filas[y_pos].append(w)
            
            # Procesar renglones
            for y in sorted(filas.keys()):
                fila_words = sorted(filas[y], key=lambda x: x['x0'])
                
                if not fila_words: continue
                
                # 1. Validar Fecha (DD/MM/AA o DD/MM/AAAA)
                texto_primera = fila_words[0]['text']
                if not re.match(r'\d{2}/\d{2}/\d{2}', texto_primera):
                    continue 

                fecha = texto_primera
                
                # 2. Buscar candidatos a importes
                candidatos_num = [w for w in fila_words if patron_numero.match(w['text'])]
                
                if not candidatos_num:
                    continue

                # LÓGICA DE SALDO vs IMPORTE
                item_importe = None
                
                if len(candidatos_num) >= 2:
                    # El último es saldo, el anteúltimo es el importe
                    item_importe = candidatos_num[-2]
                elif len(candidatos_num) == 1:
                    item_importe = candidatos_num[0]
                    # Seguridad: Si el único número está MUY a la derecha (zona de saldo), lo ignoramos
                    # Asumimos que el saldo suele estar más allá de X=520 en A4
                    if item_importe['x0'] > 530: 
                        continue

                if item_importe is None:
                    continue

                # 3. Convertir a float
                try:
                    valor_str = item_importe['text'].replace('.', '').replace(',', '.')
                    valor_float = float(valor_str)
                except:
                    continue

                # 4. Clasificar DEBE vs HABER usando la coordenada X
                debito = 0.0
                credito = 0.0
                
                if item_importe['x0'] < x_corte_final:
                    debito = valor_float
                else:
                    credito = valor_float

                # 5. Limpieza Descripción
                desc_words = [
                    w['text'] for w in fila_words 
                    if w != item_importe 
                    and w not in candidatos_num 
                    and w['text'] != fecha
                ]
                descripcion = " ".join(desc_words).strip()

                datos.append({
                    "Fecha": fecha,
                    "Descripción": descripcion,
                    "Débito": debito,
                    "Crédito": credito
                })

    return pd.DataFrame(datos), x_corte_final

# --- INTERFAZ ---

st.title("🏦 Conversor Credicoop PDF 2.0")

with st.sidebar:
    st.header("⚙️ Calibración")
    modo_auto = st.checkbox("📍 Auto-detectar columnas", value=True, help="Intenta buscar la palabra 'CREDITO' para ajustar solo.")
    
    st.write("---")
    st.write("**Ajuste Manual** (si falla el auto)")
    x_corte = st.slider("Límite visual (Eje X)", 300, 600, 420, disabled=modo_auto)
    
    uploaded_file = st.file_uploader("Cargar PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        df, x_usado = procesar_pdf_inteligente(uploaded_file, x_corte, modo_auto)

        if not df.empty:
            # Métricas
            col1, col2, col3 = st.columns(3)
            deb = df["Débito"].sum()
            cre = df["Crédito"].sum()
            saldo = cre - deb

            col1.metric("Total Débitos", f"${deb:,.2f}")
            col2.metric("Total Créditos", f"${cre:,.2f}")
            col3.metric("Neto Periodo", f"${saldo:,.2f}", delta_color="normal")
            
            st.info(f"ℹ️ Se usó el corte en la coordenada X: **{int(x_usado)}**. (Números a la izquierda son débitos, a la derecha créditos).")

            # Tabla
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            st.download_button("📥 Descargar Excel", output.getvalue(), "credicoop_procesado.xlsx", "application/vnd.ms-excel")
        else:
            st.warning("⚠️ No se encontraron movimientos. Intenta desactivar el 'Auto-detectar' y mueve el slider manualmente.")

    except Exception as e:
        st.error(f"Error: {e}")
