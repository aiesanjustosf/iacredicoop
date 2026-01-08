import streamlit as st
import pandas as pd
import pdfplumber
import re
import io
import os

# --- INTENTO DE IMPORTAR LIBRERÍAS DE IMAGEN (OCR) ---
try:
    import pytesseract
    from pytesseract import Output
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    pass 

# --- CONFIGURACIÓN DE PÁGINA Y FAVICON ---
st.set_page_config(
    page_title="Conciliador AIE",
    page_icon="favicon.ico", # Aquí toma tu icono
    layout="centered"
)

# --- CSS PERSONALIZADO (ESTÉTICA AIE) ---
st.markdown("""
    <style>
    /* Estilo de las Tarjetas de Métricas */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #E30613; /* Rojo Institucional */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        flex: 1;
    }
    .metric-value {
        font-size: 22px;
        font-weight: bold;
        color: #333;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Cajas de Estado (Check) */
    .status-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .success {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
    }
    .danger {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE LIMPIEZA ---

def limpiar_monto(texto):
    """Convierte '1.050,50' a float 1050.50"""
    try:
        limpio = re.sub(r'[^\d.,-]', '', texto)
        limpio = limpio.replace('.', '').replace(',', '.')
        return float(limpio)
    except:
        return 0.0

def es_numero_argentino(texto):
    # Detecta formato X.XXX,XX
    return bool(re.match(r'^-?[\d\.]+,\d{2}$', texto))

# --- PROCESAMIENTO NATIVO (TEXTO) ---
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

                # 1. SALDO ANTERIOR
                if "SALDO ANTERIOR" in texto_upper:
                    nums = [w for w in fila_words if es_numero_argentino(w['text'])]
                    if nums: saldo_inicial = limpiar_monto(nums[-1]['text'])
                    continue 

                # 2. SALDO FINAL (STOP)
                if "SALDO AL" in texto_upper:
                    nums = [w for w in fila_words if es_numero_argentino(w['text'])]
                    if nums: saldo_final_pdf = limpiar_monto(nums[-1]['text'])
                    encontrado_saldo_final = True
                    break

                # 3. MOVIMIENTOS
                match_fecha = re.match(r'\d{2}/\d{2}/\d{2}', fila_words[0]['text'])
                if not match_fecha: continue
                
                fecha = match_fecha.group(0)
                candidatos = [w for w in fila_words if es_numero_argentino(w['text'])]
                if not candidatos: continue

                item_importe = None
                if len(candidatos) >= 2:
                    item_importe = candidatos[-2]
                elif len(candidatos) == 1:
                    item_importe = candidatos[0]
                    if item_importe['x0'] > 520: continue 

                if not item_importe: continue

                monto = limpiar_monto(item_importe['text'])
                # Lógica visual pura
                debito = monto if item_importe['x0'] < x_corte else 0.0
                credito = monto if item_importe['x0'] >= x_corte else 0.0

                desc_tokens = [w['text'] for w in fila_words if w != item_importe and w not in candidatos and w['text'] != fecha]
                
                datos.append({
                    "Fecha": fecha,
                    "Descripción": " ".join(desc_tokens).strip(),
                    "Débito": debito,
                    "Crédito": credito
                })

    return pd.DataFrame(datos), saldo_inicial, saldo_final_pdf

# --- PROCESAMIENTO OCR (IMAGEN) ---
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
        
        # Agrupar lineas OCR
        lineas = {}
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:
                txt = ocr_data['text'][i].strip()
                if not txt: continue
                top = ocr_data['top'][i]
                found = False
                for t in lineas:
                    if abs(t - top) < 10: # Tolerancia vertical
                        lineas[t].append({'text': txt, 'left': ocr_data['left'][i]})
                        found = True
                        break
                if not found: lineas[top] = [{'text': txt, 'left': ocr_data['left'][i]}]

        for top in sorted(lineas.keys()):
            words = sorted(lineas[top], key=lambda x: x['left'])
            texto_linea = " ".join([w['text'] for w in words]).upper()
            
            # 1. SALDO ANTERIOR
            if "SALDO ANTERIOR" in texto_linea:
                nums = [w for w in words if es_numero_argentino(w['text'])]
                if nums: saldo_inicial = limpiar_monto(nums[-1]['text'])
                continue

            # 2. SALDO FINAL
            if "SALDO AL" in texto_linea:
                nums = [w for w in words if es_numero_argentino(w['text'])]
                if nums: saldo_final_pdf = limpiar_monto(nums[-1]['text'])
                encontrado_saldo_final = True
                break

            # 3. MOVIMIENTOS
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
            
            raw_desc = texto_linea.replace(fecha, "").replace(item_importe['text'], "")
            
            datos.append({
                "Fecha": fecha,
                "Descripción": raw_desc.strip(),
                "Débito": debito,
                "Crédito": credito
            })

    return pd.DataFrame(datos), saldo_inicial, saldo_final_pdf


# --- INTERFAZ DE USUARIO ---

with st.sidebar:
    # Cargar Logo si existe
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.warning("No se encontró 'logo.png' en el repo")
        
    st.markdown("### Procesador Bancario")
    st.info("Detecta automáticamente si es PDF de texto o imagen.")
    
    st.divider()
    st.write("**Calibración Fina**")
    x_corte_nativo = st.slider("Corte Visual (PDF Texto)", 300, 500, 400)
    x_corte_ocr = st.slider("Corte Visual (OCR %)", 0.4, 0.9, 0.65)
    
    uploaded_file = st.file_uploader("Subir Archivo (.pdf)", type=["pdf"])

st.title("Conciliación Automática Credicoop")

if uploaded_file:
    bytes_data = uploaded_file.read()
    
    # Procesar
    try:
        # 1. Intento Nativo
        df, s_ini, s_fin = procesar_nativo(bytes_data, x_corte_nativo)
        origen = "Lectura Digital (Texto)"
        
        if df.empty:
             # 2. Intento OCR
             with st.spinner("⏳ PDF escaneado detectado. Aplicando IA (esto demora unos segundos)..."):
                df, s_ini, s_fin = procesar_ocr_posicional(bytes_data, x_corte_ocr)
                origen = "Lectura OCR (Imagen)"

    except Exception as e:
        st.error(f"Error procesando archivo: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # --- CÁLCULOS ---
        t_deb = df["Débito"].sum()
        t_cre = df["Crédito"].sum()
        
        # Fórmula: Saldo Inicial + (Créditos - Débitos) = Saldo Calculado
        saldo_calc = s_ini + t_cre - t_deb
        diferencia = saldo_calc - s_fin
        concilia = abs(diferencia) < 1.00 # Tolerancia $1

        # --- TARJETAS DE CONCILIACIÓN ---
        # HTML personalizado para mostrar las 4 tarjetas en fila
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Saldo Anterior</div>
                <div class="metric-value">${s_ini:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Créditos</div>
                <div class="metric-value" style="color: #198754;">+${t_cre:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Débitos</div>
                <div class="metric-value" style="color: #dc3545;">-${t_deb:,.2f}</div>
            </div>
            <div class="metric-card" style="border-left: 5px solid #0d6efd;">
                <div class="metric-label">Saldo Calculado</div>
                <div class="metric-value" style="color: #0d6efd;">${saldo_calc:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- CHECK DE VALIDACIÓN ---
        if concilia:
            st.markdown(f"""
            <div class="status-box success">
                ✅ CONCILIACIÓN CORRECTA <br>
                El saldo calculado coincide con el resumen (${s_fin:,.2f})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-box danger">
                ⚠️ ERROR DE CONCILIACIÓN <br>
                Saldo PDF: ${s_fin:,.2f} vs Calculado: ${saldo_calc:,.2f} <br>
                Diferencia: ${diferencia:,.2f}
            </div>
            """, unsafe_allow_html=True)
            
        st.caption(f"Tecnología utilizada: {origen}")

        # --- TABLA Y EXCEL ---
        st.subheader("Movimientos Detallados")
        # Formato visual
        df_show = df.copy()
        df_show['Débito'] = df_show['Débito'].apply(lambda x: f"{x:,.2f}" if x > 0 else "-")
        df_show['Crédito'] = df_show['Crédito'].apply(lambda x: f"{x:,.2f}" if x > 0 else "-")
        
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Generar Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Hoja 1: Resumen
            resumen = pd.DataFrame({
                'Concepto': ['Saldo Anterior', 'Créditos', 'Débitos', 'Saldo Calculado', 'Saldo PDF', 'Diferencia'],
                'Importe': [s_ini, t_cre, t_deb, saldo_calc, s_fin, diferencia]
            })
            resumen.to_excel(writer, sheet_name='Conciliacion', index=False)
            # Hoja 2: Datos
            df.to_excel(writer, sheet_name='Movimientos', index=False)
            
        st.download_button("📥 Descargar Excel Completo", buffer.getvalue(), "conciliacion_aie.xlsx")

    else:
        st.warning("⚠️ No se pudieron extraer datos. Revisa la calibración en la barra lateral.")
