import streamlit as st
import pandas as pd
import pdfplumber
import re
import io

# --- INTENTO DE IMPORTAR LIBRERÍAS DE IMAGEN ---
try:
    import pytesseract
    from pytesseract import Output
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    pass # Se maneja el error en el cuerpo si faltan

# --- CONFIGURACIÓN E IMAGEN AIE ---
# Ajusta 'favicon.png' al nombre real de tu archivo en el repo si es distinto
st.set_page_config(
    page_title="Conciliador Bancario AIE",
    page_icon="📊", 
    layout="centered"
)

# --- CSS AIE ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #E30613; /* Rojo AIE aproximado */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-weight: bold;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---

def limpiar_monto(texto):
    """Convierte '1.050,50' a float 1050.50"""
    try:
        limpio = re.sub(r'[^\d.,-]', '', texto)
        limpio = limpio.replace('.', '').replace(',', '.')
        return float(limpio)
    except:
        return 0.0

def es_numero_argentino(texto):
    return bool(re.match(r'^-?[\d\.]+,\d{2}$', texto))

# --- ESTRUCTURA DE RETORNO ---
# Ahora las funciones devolverán: (DataFrame, SaldoInicial, SaldoFinalPDF)

# --- MOTOR 1: PDF NATIVO ---
def procesar_nativo(pdf_bytes, x_corte):
    datos = []
    saldo_inicial = 0.0
    saldo_final_pdf = 0.0
    encontrado_saldo_final = False

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            if encontrado_saldo_final: break

            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            filas = {}
            for w in words:
                y = round(w['top'])
                if y not in filas: filas[y] = []
                filas[y].append(w)
            
            for y in sorted(filas.keys()):
                fila_words = sorted(filas[y], key=lambda x: x['x0'])
                texto_completo = " ".join([w['text'] for w in fila_words])
                texto_upper = texto_completo.upper()

                # --- 1. DETECTAR SALDO ANTERIOR ---
                if "SALDO ANTERIOR" in texto_upper:
                    nums = [w for w in fila_words if es_numero_argentino(w['text'])]
                    if nums:
                        saldo_inicial = limpiar_monto(nums[-1]['text'])
                    continue # No es un movimiento, saltamos

                # --- 2. DETECTAR SALDO FINAL (STOP) ---
                if "SALDO AL" in texto_upper:
                    nums = [w for w in fila_words if es_numero_argentino(w['text'])]
                    if nums:
                        saldo_final_pdf = limpiar_monto(nums[-1]['text'])
                    encontrado_saldo_final = True
                    break

                # --- 3. PROCESAR MOVIMIENTOS ---
                match_fecha = re.match(r'\d{2}/\d{2}/\d{2}', fila_words[0]['text'])
                if not match_fecha: continue
                
                fecha = match_fecha.group(0)
                candidatos_num = [w for w in fila_words if es_numero_argentino(w['text'])]
                if not candidatos_num: continue

                item_importe = None
                if len(candidatos_num) >= 2:
                    item_importe = candidatos_num[-2]
                elif len(candidatos_num) == 1:
                    item_importe = candidatos_num[0]
                    if item_importe['x0'] > 520: continue # Zona saldo

                if not item_importe: continue

                monto = limpiar_monto(item_importe['text'])
                debito = monto if item_importe['x0'] < x_corte else 0.0
                credito = monto if item_importe['x0'] >= x_corte else 0.0

                # Limpieza descripción
                desc_tokens = [w['text'] for w in fila_words if w != item_importe and w not in candidatos_num and w['text'] != fecha]
                descripcion = " ".join(desc_tokens).strip()

                datos.append({
                    "Fecha": fecha,
                    "Descripción": descripcion,
                    "Débito": debito,
                    "Crédito": credito
                })

    return pd.DataFrame(datos), saldo_inicial, saldo_final_pdf

# --- MOTOR 2: OCR POSICIONAL ---
def procesar_ocr_posicional(pdf_bytes, x_corte_relativo=0.65):
    try:
        images = convert_from_bytes(pdf_bytes)
    except:
        return pd.DataFrame(), 0.0, 0.0

    datos = []
    saldo_inicial = 0.0
    saldo_final_pdf = 0.0
    encontrado_saldo_final = False
    
    custom_config = r'--oem 3 --psm 6'

    for img in images:
        if encontrado_saldo_final: break
        width, height = img.size
        ocr_data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT, lang='spa')
        
        # Agrupar lineas
        lineas = {}
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:
                txt = ocr_data['text'][i].strip()
                if not txt: continue
                top = ocr_data['top'][i]
                found = False
                for t in lineas:
                    if abs(t - top) < 10:
                        lineas[t].append({'text': txt, 'left': ocr_data['left'][i]})
                        found = True
                        break
                if not found: lineas[top] = [{'text': txt, 'left': ocr_data['left'][i]}]

        for top in sorted(lineas.keys()):
            words = sorted(lineas[top], key=lambda x: x['left'])
            texto_linea = " ".join([w['text'] for w in words]).upper()
            
            # --- 1. SALDO ANTERIOR ---
            if "SALDO ANTERIOR" in texto_linea:
                nums = [w for w in words if es_numero_argentino(w['text'])]
                if nums: saldo_inicial = limpiar_monto(nums[-1]['text'])
                continue

            # --- 2. SALDO FINAL ---
            if "SALDO AL" in texto_linea:
                nums = [w for w in words if es_numero_argentino(w['text'])]
                if nums: saldo_final_pdf = limpiar_monto(nums[-1]['text'])
                encontrado_saldo_final = True
                break

            # --- 3. MOVIMIENTOS ---
            match_fecha = re.match(r'\d{2}/\d{2}/\d{2}', words[0]['text'])
            if not match_fecha: continue
            
            fecha = match_fecha.group(0)
            candidatos = [w for w in words if es_numero_argentino(w['text'])]
            if not candidatos: continue

            item_importe = None
            if len(candidatos) >= 2:
                item_importe = candidatos[-2]
            elif len(candidatos) == 1:
                item_importe = candidatos[0]
                if (item_importe['left'] / width) > 0.85: continue

            if not item_importe: continue
            
            monto = limpiar_monto(item_importe['text'])
            pos_rel = item_importe['left'] / width
            
            debito = monto if pos_rel < x_corte_relativo else 0.0
            credito = monto if pos_rel >= x_corte_relativo else 0.0
            
            # Descripción sucia (OCR es dificil limpiar perfecto sin borrar de más)
            raw_desc = texto_linea.replace(fecha, "").replace(item_importe['text'], "")
            
            datos.append({
                "Fecha": fecha,
                "Descripción": raw_desc.strip(),
                "Débito": debito,
                "Crédito": credito
            })

    return pd.DataFrame(datos), saldo_inicial, saldo_final_pdf


# --- INTERFAZ AIE ---

with st.sidebar:
    # LOGO AIE
    # Si tienes el archivo en el root de tu repo, descomenta esto:
    # st.image("logo.png", use_column_width=True) 
    st.markdown("### 🏢 AIE San Justo")
    
    st.divider()
    st.header("⚙️ Calibración")
    x_corte_nativo = st.slider("Corte Visual (PDF)", 300, 500, 400)
    x_corte_ocr = st.slider("Corte Visual (OCR %)", 0.4, 0.9, 0.65)
    
    st.divider()
    uploaded_file = st.file_uploader("Subir Resumen Credicoop", type=["pdf"])

st.title("Conciliador Automático")

if uploaded_file:
    bytes_data = uploaded_file.read()
    
    # Procesar
    try:
        # Intento 1: Nativo
        df, s_ini, s_fin = procesar_nativo(bytes_data, x_corte_nativo)
        tipo = "Nativo (Texto)"
        
        if df.empty:
             # Intento 2: OCR
             with st.spinner("Leyendo escaneo con IA..."):
                df, s_ini, s_fin = procesar_ocr_posicional(bytes_data, x_corte_ocr)
                tipo = "OCR (Imagen)"

    except Exception as e:
        st.error(f"Error crítico: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # --- CÁLCULOS DE CONCILIACIÓN ---
        tot_deb = df["Débito"].sum()
        tot_cre = df["Crédito"].sum()
        
        # Fórmula Bancaria: Saldo Anterior + Créditos - Débitos = Nuevo Saldo
        saldo_calculado = s_ini + tot_cre - tot_deb
        diferencia = saldo_calculado - s_fin
        
        match_ok = abs(diferencia) < 1.00 # Tolerancia de $1 por redondeos

        # --- MOSTRAR MÉTRICAS (TARJETAS) ---
        c1, c2, c3, c4 = st.columns(4)
        
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Saldo Anterior</div><div class="metric-value">${s_ini:,.2f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Total Créditos</div><div class="metric-value" style="color:green">+${tot_cre:,.2f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Total Débitos</div><div class="metric-value" style="color:red">-${tot_deb:,.2f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Saldo Calculado</div><div class="metric-value" style="color:blue">${saldo_calculado:,.2f}</div></div>', unsafe_allow_html=True)

        st.divider()

        # --- VERIFICACIÓN (CHECK) ---
        col_check_1, col_check_2 = st.columns([3, 1])
        
        with col_check_1:
            if match_ok:
                st.markdown(f"""
                <div class="success-box">
                    ✅ CONCILIACIÓN EXITOSA<br>
                    El saldo calculado coincide con el "Saldo al" del PDF (${s_fin:,.2f})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-box">
                    ⚠️ DIFERENCIA DETECTADA<br>
                    Saldo PDF: ${s_fin:,.2f} vs Calculado: ${saldo_calculado:,.2f}<br>
                    Diferencia: ${diferencia:,.2f}
                </div>
                """, unsafe_allow_html=True)
        
        with col_check_2:
             st.caption(f"Modo: {tipo}")

        # --- TABLA ---
        st.subheader("Detalle de Movimientos")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # --- EXCEL ---
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Hoja Resumen
            resumen_df = pd.DataFrame([
                {"Concepto": "Saldo Anterior", "Importe": s_ini},
                {"Concepto": "Total Créditos", "Importe": tot_cre},
                {"Concepto": "Total Débitos", "Importe": tot_deb},
                {"Concepto": "Saldo Final Calc", "Importe": saldo_calculado},
                {"Concepto": "Saldo PDF", "Importe": s_fin},
                {"Concepto": "Diferencia", "Importe": diferencia}
            ])
            resumen_df.to_excel(writer, sheet_name="Resumen", index=False)
            df.to_excel(writer, sheet_name="Movimientos", index=False)
            
        st.download_button("📥 Descargar Excel Conciliado", buffer.getvalue(), "conciliacion_aie.xlsx")

    else:
        st.warning("No se encontraron movimientos o no se pudo leer el archivo.")
