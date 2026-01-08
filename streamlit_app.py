import streamlit as st
import pandas as pd
import pdfplumber
import re
import io

# --- CONFIGURACIÓN DE PÁGINA (ESTÉTICA Y ANCHO) ---
st.set_page_config(
    page_title="Procesador Credicoop IA",
    page_icon="🏦",
    layout="centered", # ESTO EVITA QUE OCUPE TODO EL ANCHO
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO PARA ESTÉTICA ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #005f9e;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1 {
        color: #004481;
        font-family: 'Helvetica', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIÓN DE PROCESAMIENTO (CORE LÓGICO) ---
def procesar_pdf_credicoop_posicional(pdf_file, x_corte):
    """
    Procesa el PDF agrupando palabras por coordenadas.
    x_corte: Coordenada X que divide visualmente Débitos de Créditos.
    """
    datos = []
    # Regex estricta para números argentinos: 1.000,00 o -50,00
    patron_numero = re.compile(r'^-?[\d\.]+,\d{2}$') 

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Extraer palabras con sus coordenadas (x0, top, text)
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            # Agrupar palabras por renglón (usando 'top' redondeado)
            filas = {}
            for w in words:
                y_pos = round(w['top']) # Redondeo para agrupar misma linea
                if y_pos not in filas:
                    filas[y_pos] = []
                filas[y_pos].append(w)
            
            # Procesar cada renglón ordenado verticalmente
            for y in sorted(filas.keys()):
                fila_words = sorted(filas[y], key=lambda x: x['x0']) # Ordenar de izq a der
                
                # 1. Validar si empieza con FECHA
                if not fila_words: continue
                texto_primera = fila_words[0]['text']
                if not re.match(r'\d{2}/\d{2}/\d{2}', texto_primera):
                    continue # Si no arranca con fecha, es basura (encabezado, saldo anterior, pie)

                fecha = texto_primera
                
                # 2. Identificar números monetarios en la fila
                candidatos_num = [w for w in fila_words if patron_numero.match(w['text'])]
                
                if not candidatos_num:
                    continue

                # --- LÓGICA DE NEGOCIO INDICADA ---
                # Si hay 2 números: Último = Saldo (Ignorar), Penúltimo = Importe.
                # Si hay 1 número: Es el Importe.
                
                item_importe = None
                
                if len(candidatos_num) >= 2:
                    item_importe = candidatos_num[-2] # El anteúltimo
                elif len(candidatos_num) == 1:
                    item_importe = candidatos_num[0]
                    # Check de seguridad: si está DEMASIADO a la derecha (>500), podría ser solo una linea de saldo
                    if item_importe['x0'] > 530: 
                        continue

                if item_importe is None:
                    continue

                # 3. Parsear valor
                try:
                    valor_str = item_importe['text'].replace('.', '').replace(',', '.')
                    valor_float = float(valor_str)
                except:
                    continue

                # 4. Determinar Débito vs Crédito por POSICIÓN X
                debito = 0.0
                credito = 0.0
                
                # Aquí usamos el calibrador X_CORTE
                if item_importe['x0'] < x_corte:
                    debito = valor_float
                else:
                    credito = valor_float

                # 5. Limpiar Descripción
                # Unimos todo lo que NO es la fecha ni los números detectados
                desc_words = [
                    w['text'] for w in fila_words 
                    if w != item_importe 
                    and w not in candidatos_num # Saca también el saldo si existía
                    and w['text'] != fecha
                ]
                descripcion = " ".join(desc_words).strip()

                datos.append({
                    "Fecha": fecha,
                    "Descripción": descripcion,
                    "Débito": debito,
                    "Crédito": credito
                })

    return pd.DataFrame(datos)

# --- INTERFAZ DE USUARIO ---

st.title("🏦 Conversor Credicoop PDF")
st.markdown("Sube tu resumen en PDF. El sistema detectará las columnas por posición visual.")

with st.sidebar:
    st.header("⚙️ Calibración")
    st.info("Si los débitos caen en créditos (o viceversa), ajusta este control.")
    # El valor 420 suele funcionar bien para A4 vertical estándar de bancos
    x_corte = st.slider("Límite visual (Eje X)", min_value=300, max_value=600, value=420, step=5, 
                        help="Coordenada que separa la columna Debe de la columna Haber.")
    
    uploaded_file = st.file_uploader("Cargar PDF Credicoop", type=["pdf"])

if uploaded_file is not None:
    try:
        # Procesar
        df = procesar_pdf_credicoop_posicional(uploaded_file, x_corte)

        if not df.empty:
            # --- TARJETAS DE TOTALES ---
            col1, col2, col3 = st.columns(3)
            total_debito = df["Débito"].sum()
            total_credito = df["Crédito"].sum()
            saldo_periodo = total_credito - total_debito

            with col1:
                st.markdown(f'<div class="metric-card"><h4>Total Débitos</h4><h2 style="color: #d9534f;">${total_debito:,.2f}</h2></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h4>Total Créditos</h4><h2 style="color: #5cb85c;">${total_credito:,.2f}</h2></div>', unsafe_allow_html=True)
            with col3:
                color_saldo = "#5cb85c" if saldo_periodo >= 0 else "#d9534f"
                st.markdown(f'<div class="metric-card"><h4>Neto Periodo</h4><h2 style="color: {color_saldo};">${saldo_periodo:,.2f}</h2></div>', unsafe_allow_html=True)

            st.divider()

            # --- TABLA DE DATOS ---
            st.subheader("📝 Detalle de Movimientos")
            
            # Formato para visualización
            df_display = df.copy()
            df_display["Débito"] = df_display["Débito"].apply(lambda x: f"{x:,.2f}" if x > 0 else "")
            df_display["Crédito"] = df_display["Crédito"].apply(lambda x: f"{x:,.2f}" if x > 0 else "")

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # --- DESCARGA ---
            st.divider()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Credicoop')
            
            st.download_button(
                label="📥 Descargar Excel",
                data=output.getvalue(),
                file_name="movimientos_credicoop.xlsx",
                mime="application/vnd.ms-excel"
            )

        else:
            st.warning("No se encontraron movimientos. Verifica si el PDF es legible o ajusta la calibración.")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
else:
    st.info("👆 Carga un archivo en el menú lateral para comenzar.")
