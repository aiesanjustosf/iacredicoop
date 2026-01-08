import streamlit as st
import pandas as pd
import pdfplumber
import re
import io

# Intentamos importar librerías de OCR y procesamiento de imagen
try:
    import pytesseract
    from pytesseract import Output
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    st.error("⚠️ Faltan librerías. Asegúrate de que requirements.txt tenga: pytesseract, pdf2image, pillow, opencv-python-headless")

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Procesador Credicoop V3", layout="centered")

# --- FUNCIONES AUXILIARES ---

def limpiar_monto(texto):
    """
    Convierte formato argentino '1.050,50' a float 1050.50
    """
    try:
        # Eliminar cualquier caracter que no sea número, punto, coma o signo menos
        limpio = re.sub(r'[^\d.,-]', '', texto)
        # Reemplazar puntos de miles por nada y coma decimal por punto
        limpio = limpio.replace('.', '').replace(',', '.')
        return float(limpio)
    except:
        return 0.0

def es_numero_argentino(texto):
    # Regex estricta: números con decimales ,XX al final
    return bool(re.match(r'^-?[\d\.]+,\d{2}$', texto))

# --- MOTOR 1: PDF NATIVO (Texto Seleccionable) ---
def procesar_nativo(pdf_bytes, x_corte):
    datos = []
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # Extracción de palabras con coordenadas
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            # Agrupar por filas (eje Y)
            filas = {}
            for w in words:
                y = round(w['top'])
                if y not in filas: filas[y] = []
                filas[y].append(w)
            
            # Procesar fila por fila
            for y in sorted(filas.keys()):
                fila_words = sorted(filas[y], key=lambda x: x['x0'])
                texto_completo = " ".join([w['text'] for w in fila_words])
                
                # --- CONDICIÓN DE CORTE (STOP) ---
                if "SALDO AL" in texto_completo.upper():
                    # Aquí termina el resumen de movimientos
                    return pd.DataFrame(datos)

                # --- DETECCIÓN SALDO ANTERIOR ---
                if "SALDO ANTERIOR" in texto_completo.upper():
                    # Buscamos el último número de la línea
                    nums = [w for w in fila_words if es_numero_argentino(w['text'])]
                    if nums:
                        saldo_inicial = limpiar_monto(nums[-1]['text'])
                        datos.append({
                            "Fecha": "Inicio",
                            "Descripción": "SALDO ANTERIOR",
                            "Débito": 0.0,
                            "Crédito": 0.0,
                            "Saldo": saldo_inicial, # Columna extra informativa
                            "Origen": "Nativo"
                        })
                    continue

                # --- FILTRO DE FECHA ---
                # Si la linea no arranca con fecha, la ignoramos (salvo que fuera saldo anterior procesado arriba)
                # Buscamos patrón DD/MM/AA al inicio
                primera_palabra = fila_words[0]['text']
                match_fecha = re.match(r'\d{2}/\d{2}/\d{2}', primera_palabra)
                
                if not match_fecha:
                    continue
                
                fecha = match_fecha.group(0)

                # --- LÓGICA DE IMPORTES ---
                # Buscamos todas las "palabras" que parecen dinero
                candidatos_num = [w for w in fila_words if es_numero_argentino(w['text'])]
                
                if not candidatos_num:
                    continue

                item_importe = None
                
                # REGLA: Si hay 2 números, el de la derecha es Saldo (Ignorar), el anterior es Importe.
                if len(candidatos_num) >= 2:
                    item_importe = candidatos_num[-2]
                elif len(candidatos_num) == 1:
                    item_importe = candidatos_num[0]
                    # Seguridad: Si está muy a la derecha (zona saldo), ignorar.
                    # Asumiendo ancho de página ~595pt (A4), el saldo suele estar > 520
                    if item_importe['x0'] > 520:
                        continue

                if not item_importe:
                    continue

                # Determinar Columna por posición X
                monto = limpiar_monto(item_importe['text'])
                debito = 0.0
                credito = 0.0

                if item_importe['x0'] < x_corte:
                    debito = monto
                else:
                    credito = monto

                # Limpieza Descripción
                desc_tokens = [
                    w['text'] for w in fila_words 
                    if w != item_importe and w not in candidatos_num and w['text'] != fecha
                ]
                descripcion = " ".join(desc_tokens).strip()

                datos.append({
                    "Fecha": fecha,
                    "Descripción": descripcion,
                    "Débito": debito,
                    "Crédito": credito,
                    "Origen": "Nativo"
                })

    return pd.DataFrame(datos)


# --- MOTOR 2: OCR POSICIONAL (Para imágenes) ---
def procesar_ocr_posicional(pdf_bytes, x_corte_relativo=0.65):
    """
    Convierte PDF a imagen y usa Tesseract para obtener DATA (con coordenadas),
    no solo string. Así podemos separar Debe/Haber visualmente.
    x_corte_relativo: % del ancho de la página donde corta débito/crédito.
    """
    try:
        images = convert_from_bytes(pdf_bytes)
    except:
        return pd.DataFrame()

    datos = []
    
    # Configuración para detectar líneas tabularmente
    custom_config = r'--oem 3 --psm 6'

    for img in images:
        width, height = img.size
        # Obtenemos datos detallados (texto, left, top, width...)
        ocr_data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT, lang='spa')
        
        n_boxes = len(ocr_data['text'])
        
        # Agrupar por líneas (usando 'top' con margen de error)
        # Tesseract a veces varía el 'top' por pixeles en la misma linea
        lineas = {}
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 0: # Solo confianza > 0
                texto = ocr_data['text'][i].strip()
                if not texto: continue
                
                top = ocr_data['top'][i]
                # Agrupamiento aproximado (margen de 10px)
                found_line = False
                for t in lineas.keys():
                    if abs(t - top) < 10:
                        lineas[t].append({
                            'text': texto,
                            'left': ocr_data['left'][i],
                            'width': ocr_data['width'][i]
                        })
                        found_line = True
                        break
                if not found_line:
                    lineas[top] = [{
                        'text': texto,
                        'left': ocr_data['left'][i],
                        'width': ocr_data['width'][i]
                    }]

        # Procesar líneas ordenadas
        for top in sorted(lineas.keys()):
            words = sorted(lineas[top], key=lambda x: x['left'])
            texto_linea = " ".join([w['text'] for w in words])
            
            # --- CONDICIÓN DE CORTE OCR ---
            if "SALDO AL" in texto_linea.upper():
                return pd.DataFrame(datos)

            # --- Detección Fecha ---
            match_fecha = re.match(r'\d{2}/\d{2}/\d{2}', words[0]['text'])
            if not match_fecha:
                # Chequear si es Saldo Anterior
                if "SALDO" in words[0]['text'].upper() and "ANTERIOR" in texto_linea.upper():
                     # Lógica simplificada para saldo anterior en OCR
                     pass
                continue
            
            fecha = match_fecha.group(0)

            # --- Identificar números ---
            candidatos = []
            for w in words:
                if es_numero_argentino(w['text']):
                    candidatos.append(w)
            
            if not candidatos: continue

            # Lógica Importe vs Saldo (igual que Nativo)
            item_importe = None
            if len(candidatos) >= 2:
                item_importe = candidatos[-2]
            elif len(candidatos) == 1:
                item_importe = candidatos[0]
                # Si está muy a la derecha (más del 85% del ancho), es saldo
                if (item_importe['left'] / width) > 0.85:
                    continue
            
            if not item_importe: continue

            monto = limpiar_monto(item_importe['text'])
            
            # --- DECISIÓN DEBITO VS CREDITO (VISUAL) ---
            # Calculamos la posición relativa del número en la página (0.0 a 1.0)
            pos_x_rel = item_importe['left'] / width
            
            debito = 0.0
            credito = 0.0
            
            # Aquí está la magia: Usamos el corte relativo
            # Ajuste empírico: En resúmenes Credicoop, el Crédito suele empezar pasado el 60-65%
            if pos_x_rel < x_corte_relativo:
                debito = monto
            else:
                credito = monto
            
            # Limpiar descripción
            desc_txt = texto_linea.replace(item_importe['text'], '').replace(fecha, '')
            if len(candidatos) >= 2:
                desc_txt = desc_txt.replace(candidatos[-1]['text'], '') # Quitar saldo
            
            datos.append({
                "Fecha": fecha,
                "Descripción": desc_txt.strip(),
                "Débito": debito,
                "Crédito": credito,
                "Origen": "OCR Posicional"
            })

    return pd.DataFrame(datos)

# --- INTERFAZ ---

st.title("🏦 Procesador Credicoop V3 (Inteligente)")

with st.sidebar:
    st.info("Sistema Dual: Detecta texto o usa visión por computadora.")
    
    st.write("### Calibración Nativa")
    x_corte_nativo = st.slider("Corte X (PDF Texto)", 300, 550, 400, help="Valor por defecto: 400")
    
    st.write("### Calibración OCR")
    x_corte_ocr = st.slider("Corte % Ancho (Imagen)", 0.4, 0.9, 0.68, help="Porcentaje del ancho donde empieza la columna Crédito")

    uploaded_file = st.file_uploader("Subir Resumen", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    df = pd.DataFrame()
    
    # 1. Intentar modo nativo primero
    try:
        df = procesar_nativo(file_bytes, x_corte_nativo)
        if df.empty:
            raise Exception("Sin datos textuales")
        st.success("✅ Procesado modo TEXTO DIRECTO")
        
    except Exception as e:
        # 2. Fallback a OCR Posicional
        st.warning("⚠️ No se detectó texto seleccionable. Activando escaneo visual (OCR)...")
        with st.spinner("Analizando píxeles y coordenadas..."):
            df = procesar_ocr_posicional(file_bytes, x_corte_ocr)
            if not df.empty:
                st.success("✅ Procesado modo OCR POSICIONAL")

    if not df.empty:
        # Mostrar tabla
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Totales
        col1, col2, col3 = st.columns(3)
        total_deb = df["Débito"].sum()
        total_cre = df["Crédito"].sum()
        saldo_movs = total_cre - total_deb
        
        col1.metric("Total Débitos", f"${total_deb:,.2f}")
        col2.metric("Total Créditos", f"${total_cre:,.2f}")
        col3.metric("Neto Movimientos", f"${saldo_movs:,.2f}")

        # Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Movimientos')
        
        st.download_button("📥 Descargar Excel", buffer.getvalue(), "resumen_procesado.xlsx")
    else:
        st.error("❌ No se pudieron extraer datos. Verifica que el archivo no esté dañado.")
