import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
import os

# --- INTENTO DE IMPORTAR LIBRERÍAS DE IMAGEN (OCR) ---
try:
    import pytesseract
    from pytesseract import Output
    from pdf2image import convert_from_bytes
    from PIL import Image
    TIENE_OCR = True
except ImportError:
    TIENE_OCR = False

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Conciliador AIE Pro", layout="wide")

# --- CSS (ESTÉTICA) ---
st.markdown("""
    <style>
    .metric-container { display: flex; gap: 10px; margin-bottom: 20px; }
    .metric-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #E30613; text-align: center; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-value { font-size: 20px; font-weight: bold; color: #333; }
    .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
    .status-box { padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; margin: 10px 0; }
    .success { background-color: #d1e7dd; color: #0f5132; }
    .warning { background-color: #fff3cd; color: #856404; }
    .danger { background-color: #f8d7da; color: #842029; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE LIMPIEZA ---
def limpiar_numero_ar(valor):
    if not valor: return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    val_str = str(valor).strip()
    es_negativo = False
    if val_str.endswith("-") or (val_str.startswith("(") and val_str.endswith(")")): es_negativo = True
    val_str = re.sub(r'[^\d,.]', '', val_str)
    if not val_str: return 0.0
    try:
        val_str = val_str.replace(".", "").replace(",", ".")
        num = float(val_str)
        return -num if es_negativo else num
    except: return 0.0

def formatear_moneda_ar(valor):
    if pd.isna(valor) or valor == "": return ""
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- MOTOR 1: PDF TEXTO (PDFPLUMBER CON SLIDERS) ---
def procesar_pdf_texto(pdf_file, x_coords):
    movimientos = []
    saldo_anterior = 0.0
    x_fecha, x_desc, x_debito, x_credito = x_coords

    with pdfplumber.open(pdf_file) as pdf:
        if len(pdf.pages) > 0:
            p1 = pdf.pages[0].extract_text() or ""
            m = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", p1, re.IGNORECASE)
            if m: saldo_anterior = limpiar_numero_ar(m.group(1))

        for page in pdf.pages:
            # Estrategia explicita con sliders (La más precisa para columnas)
            lines = [0, x_fecha, x_desc, x_debito, x_credito, page.width]
            settings = {"vertical_strategy": "explicit", "explicit_vertical_lines": lines, "horizontal_strategy": "text", "intersection_y_tolerance": 5}
            table = page.extract_table(settings)
            
            if table:
                for row in table:
                    row = [c.strip() if c else "" for c in row]
                    if len(row) >= 5 and re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                        if "SALDO AL" in row[1]: continue
                        try:
                            movimientos.append({
                                "Fecha": row[0], "Descripción": row[1],
                                "Débito": limpiar_numero_ar(row[2]), "Crédito": limpiar_numero_ar(row[3]),
                                "Saldo_PDF": limpiar_numero_ar(row[4])
                            })
                        except: pass
    return pd.DataFrame(movimientos), saldo_anterior

# --- MOTOR 2: OCR (IMAGEN ESCANEADA) ---
def procesar_ocr(pdf_bytes, corte_visual_pct):
    if not TIENE_OCR: return pd.DataFrame(), 0.0
    
    try: images = convert_from_bytes(pdf_bytes)
    except: return pd.DataFrame(), 0.0

    datos = []
    s_ini = 0.0
    custom_config = r'--oem 3 --psm 6'

    for img in images:
        width, height = img.size
        data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT, lang='spa')
        
        # Agrupar por líneas
        lineas = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0 and data['text'][i].strip():
                top = data['top'][i]
                found = False
                for t in lineas:
                    if abs(t - top) < 10:
                        lineas[t].append({'text': data['text'][i], 'left': data['left'][i]})
                        found = True; break
                if not found: lineas[top] = [{'text': data['text'][i], 'left': data['left'][i]}]
        
        for top in sorted(lineas.keys()):
            words = sorted(lineas[top], key=lambda x: x['left'])
            line_text = " ".join([w['text'] for w in words]).upper()
            
            if "SALDO ANTERIOR" in line_text:
                nums = [w for w in words if re.match(r'[\d.,]+', w['text'])]
                if nums: s_ini = limpiar_numero_ar(nums[-1]['text'])
                continue
                
            if re.match(r'\d{2}/\d{2}/\d{2}', words[0]['text']):
                fecha = words[0]['text']
                numeros = [w for w in words if re.match(r'[\d.,]+', w['text']) and len(w['text']) > 3]
                
                if numeros:
                    # El último número suele ser el saldo, el anteúltimo el importe
                    # Simplificación para OCR: Tomamos el importe por posición
                    item_importe = numeros[0] if len(numeros)==1 else numeros[-2]
                    monto = limpiar_numero_ar(item_importe['text'])
                    
                    pos_rel = item_importe['left'] / width
                    deb = monto if pos_rel < corte_visual_pct else 0.0
                    cred = monto if pos_rel >= corte_visual_pct else 0.0
                    
                    # Intentar buscar saldo en el último numero de la derecha
                    saldo_pdf = 0.0
                    if len(numeros) >= 2 and (numeros[-1]['left'] / width) > 0.8:
                        saldo_pdf = limpiar_numero_ar(numeros[-1]['text'])

                    datos.append({
                        "Fecha": fecha, "Descripción": " ".join([w['text'] for w in words if w!=item_importe][1:]),
                        "Débito": deb, "Crédito": cred, "Saldo_PDF": saldo_pdf
                    })
                    
    return pd.DataFrame(datos), s_ini

# --- INTELIGENCIA: AUTOCORRECCIÓN ---
def autocorregir(df, s_inicial):
    if df.empty: return df, 0, 0
    df = df.copy()
    df['Estado'] = 'OK'; df['Saldo_Calculado'] = 0.0
    acum = s_inicial
    corregidos = 0
    
    for i, row in df.iterrows():
        deb, cred, s_pdf = row['Débito'], row['Crédito'], row['Saldo_PDF']
        teorico = acum + cred - deb
        
        if s_pdf != 0 and abs(teorico - s_pdf) > 1.0:
            # Intento corrección cruzada
            if deb > 0 and abs((acum + deb) - s_pdf) < 1.0:
                df.at[i, 'Crédito'] = deb; df.at[i, 'Débito'] = 0.0
                df.at[i, 'Estado'] = 'CORREGIDO (Era Crédito)'
                acum = s_pdf; corregidos += 1
            elif cred > 0 and abs((acum - cred) - s_pdf) < 1.0:
                df.at[i, 'Débito'] = cred; df.at[i, 'Crédito'] = 0.0
                df.at[i, 'Estado'] = 'CORREGIDO (Era Débito)'
                acum = s_pdf; corregidos += 1
            else:
                df.at[i, 'Estado'] = 'ERROR'
                acum = s_pdf # Forzar sincro
        else:
            if s_pdf != 0: acum = s_pdf
            else: acum = teorico
        
        df.at[i, 'Saldo_Calculado'] = acum
        
    return df, corregidos, acum

# --- UI ---
c1, c2 = st.columns([1, 5])
with c1:
    if os.path.exists("logo_aie.png"): st.image("logo_aie.png", width=100)
    else: st.write("🏦")
with c2: st.title("Conciliador AIE (Híbrido)")

st.markdown("---")
col_conf, col_main = st.columns([1, 3])

with col_conf:
    st.header("Ajustes")
    modo = st.radio("Tecnología", ["Automático", "Texto (Sliders)", "OCR (Imagen)"])
    
    if modo == "Texto (Sliders)" or modo == "Automático":
        st.caption("Ajuste Columnas (PDF Digital)")
        x_fecha = st.slider("Fecha", 0, 100, 60)
        x_desc = st.slider("Descripción", 100, 500, 310)
        x_deb = st.slider("Débito", 300, 600, 480)
        x_cred = st.slider("Crédito", 400, 700, 580)
        
    if modo == "OCR (Imagen)" or modo == "Automático":
        st.divider()
        st.caption("Ajuste Corte Visual (Escaneados)")
        ocr_pct = st.slider("Punto de corte %", 0.4, 0.9, 0.65)

    uploaded_file = st.file_uploader("Subir PDF", type="pdf")

if uploaded_file:
    # LÓGICA DE SELECCIÓN DE MOTOR
    df_res = pd.DataFrame()
    s_ini = 0.0
    origen_datos = ""
    
    # 1. Intento Texto
    if modo in ["Automático", "Texto (Sliders)"]:
        coords = [x_fecha, x_desc, x_deb, x_cred]
        df_res, s_ini = procesar_pdf_texto(uploaded_file, coords)
        origen_datos = "Texto Digital"

    # 2. Intento OCR (si Texto falló o si se eligió OCR)
    if (df_res.empty and modo == "Automático") or modo == "OCR (Imagen)":
        if TIENE_OCR:
            with st.spinner("Aplicando OCR a imagen escaneada..."):
                uploaded_file.seek(0)
                df_res, s_ini = procesar_ocr(uploaded_file.read(), ocr_pct)
                origen_datos = "OCR (Imagen)"
        elif modo == "OCR (Imagen)":
            st.error("Librerías de OCR no instaladas en el servidor.")

    # INPUT SALDO
    with col_conf:
        st.divider()
        saldo_inicial = st.number_input("Saldo Anterior", value=s_ini, step=1000.0)

    if df_res.empty:
        st.warning("⚠️ No se extrajeron datos. Probá cambiar el modo o ajustar sliders.")
    else:
        # APLICAR INTELIGENCIA
        df_final, n_correg, s_final = autocorregir(df_res, saldo_inicial)
        
        t_deb = df_final["Débito"].sum()
        t_cred = df_final["Crédito"].sum()
        
        with col_main:
            st.info(f"Modo utilizado: {origen_datos}")
            
            # Tarjetas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Saldo Ant.", formatear_moneda_ar(saldo_inicial))
            m2.metric("Créditos", formatear_moneda_ar(t_cred))
            m3.metric("Débitos", formatear_moneda_ar(t_deb))
            m4.metric("Saldo Final", formatear_moneda_ar(s_final))
            
            if n_correg > 0:
                st.markdown(f'<div class="status-box warning">✨ {n_correg} correcciones automáticas aplicadas</div>', unsafe_allow_html=True)
            
            # Tabla
            def color_rows(row):
                if 'CORREGIDO' in row['Estado']: return ['background-color: #fff3cd']*len(row)
                if 'ERROR' in row['Estado']: return ['background-color: #f8d7da']*len(row)
                return ['']*len(row)
                
            cols_ver = ['Débito', 'Crédito', 'Saldo_PDF', 'Saldo_Calculado']
            df_view = df_final.copy()
            for c in cols_ver: df_view[c] = df_view[c].apply(formatear_moneda_ar)
            
            st.dataframe(df_view.style.apply(color_rows, axis=1), use_container_width=True, height=500)
            
            # Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_final.to_excel(writer, index=False)
            st.download_button("📥 Descargar Excel", buffer.getvalue(), "conciliacion.xlsx")
