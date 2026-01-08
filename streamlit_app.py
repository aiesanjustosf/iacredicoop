import streamlit as st
import pandas as pd
import pdfplumber
import re
import io
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    st.error("Faltan librerías. Asegúrate de actualizar requirements.txt y packages.txt")

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Procesador Credicoop Híbrido", layout="centered")

# --- LÓGICA DE PROCESAMIENTO ---

def limpiar_numero_argentino(texto):
    """Convierte '1.050,50' a float 1050.50"""
    try:
        # Quitamos puntos de miles y cambiamos coma decimal por punto
        clean = texto.replace('.', '').replace(',', '.')
        return float(clean)
    except:
        return 0.0

def parsear_linea(texto_linea, x_corte):
    """
    Analiza una línea de texto (ya sea de OCR o PDF nativo) 
    y busca patrón de fecha y números.
    """
    # Regex fecha DD/MM/AA
    match_fecha = re.search(r'(\d{2}/\d{2}/\d{2})', texto_linea)
    if not match_fecha:
        return None

    fecha = match_fecha.group(1)
    
    # Regex números (busca formatos tipo 100,00 o 1.000,00 al final de la linea)
    # Buscamos todos los importes posibles en la linea
    numeros = re.findall(r'(-?[\d\.]+,\d{2})', texto_linea)
    
    if not numeros:
        return None

    monto_debito = 0.0
    monto_credito = 0.0
    descripcion = texto_linea

    # LÓGICA DE SALDO vs IMPORTE
    importe_str = ""
    
    if len(numeros) >= 2:
        # Si hay 2 números, el último suele ser saldo, el anteúltimo el importe
        importe_str = numeros[-2]
        saldo_str = numeros[-1]
        # Limpiamos descripción quitando los números
        descripcion = descripcion.replace(importe_str, '').replace(saldo_str, '')
    elif len(numeros) == 1:
        importe_str = numeros[0]
        descripcion = descripcion.replace(importe_str, '')

    valor = limpiar_numero_argentino(importe_str)

    # AQUÍ ESTÁ EL TRUCO PARA EL DEBITO/CREDITO
    # Como en OCR perdemos la coordenada exacta X, usamos heurística de texto o
    # intentamos ver si hay muchos espacios en blanco antes del número.
    # PERO, para PDF NATIVO, usaremos coordenadas.
    # Para OCR, esta función es limitada, así que la lógica principal va en las funciones especificas.
    
    return {
        "fecha": fecha,
        "descripcion": descripcion.strip(),
        "valor_bruto": valor,
        "raw_text": texto_linea
    }

# --- ESTRATEGIA 1: PDF NATIVO (Texto seleccionable) ---
def procesar_nativo(pdf_bytes, x_corte):
    datos = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # Chequeo rápido: ¿Tiene texto esta página?
            text_check = page.extract_text()
            if not text_check or len(text_check) < 50:
                raise Exception("Página sin texto detectado (Posible imagen)")

            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            # Agrupar por filas
            filas = {}
            for w in words:
                y = round(w['top'])
                if y not in filas: filas[y] = []
                filas[y].append(w)
            
            for y in sorted(filas.keys()):
                fila_words = sorted(filas[y], key=lambda x: x['x0'])
                linea_txt = " ".join([w['text'] for w in fila_words])
                
                # Pre-análisis
                parsed = parsear_linea(linea_txt, x_corte)
                if not parsed: continue

                # BUSQUEDA DEL OBJETO PALABRA QUE CORRESPONDE AL VALOR
                # Para saber su coordenada X real
                valor_obj = None
                # Buscamos la palabra que coincida con el valor string (ej "500,00")
                # Lo hacemos buscando de derecha a izquierda (reverse)
                for w in reversed(fila_words):
                    if re.match(r'^-?[\d\.]+,\d{2}$', w['text']):
                        # Si hay dos números (saldo e importe), el primero que encontramos desde la derecha es saldo (X muy grande)
                        # El segundo es el importe.
                        # Hacemos lógica simple: Si X > 530 es saldo casi seguro.
                        if w['x0'] > 530:
                            continue # Ignoramos saldo
                        valor_obj = w
                        break
                
                if valor_obj:
                    if valor_obj['x0'] < x_corte:
                        debito = parsed['valor_bruto']
                        credito = 0.0
                    else:
                        debito = 0.0
                        credito = parsed['valor_bruto']
                else:
                    # Si no pudimos matchear coordenada, fallback a lógica simple
                    # Si dice "IMPUESTO" o "PAGO" -> debito (heurística de emergencia)
                    if "IMPUESTO" in parsed['descripcion'].upper() or "COMISION" in parsed['descripcion'].upper():
                        debito = parsed['valor_bruto']; credito = 0.0
                    else:
                        # Por defecto crédito si falla todo (peligroso, pero necesario fallback)
                        debito = 0.0; credito = parsed['valor_bruto']

                datos.append({
                    "Fecha": parsed['fecha'],
                    "Descripción": parsed['descripcion'],
                    "Débito": debito,
                    "Crédito": credito,
                    "Origen": "Nativo"
                })
    return pd.DataFrame(datos)

# --- ESTRATEGIA 2: OCR (Imágenes) ---
def procesar_ocr(pdf_bytes):
    # Convertir PDF a imagenes
    try:
        images = convert_from_bytes(pdf_bytes)
    except Exception as e:
        st.error(f"Error convirtiendo PDF a imagen (Falta Poppler?): {e}")
        return pd.DataFrame()

    datos = []
    # Configuración Tesseract: Asumimos columnas espaciadas
    custom_config = r'--oem 3 --psm 6' 

    for img in images:
        # Extraer texto crudo manteniendo layout físico
        texto_crudo = pytesseract.image_to_string(img, config=custom_config, lang='spa')
        
        # Procesar linea a linea
        lines = texto_crudo.split('\n')
        for line in lines:
            if not line.strip(): continue
            
            parsed = parsear_linea(line, 0) # X corte no sirve aqui igual
            if not parsed: continue
            
            # EN OCR NO TENEMOS COORDENADAS X FIABLES FÁCILMENTE
            # Usamos lógica de espaciado o palabras clave
            # Si hay un espacio gigante entre descripción y número -> Crédito?
            # Es muy difícil. Vamos a usar heurística de palabras + posición relativa string
            
            es_credito = False
            
            # Heurística 1: Palabras clave
            desc_upper = parsed['descripcion'].upper()
            if "TRANSFERENCIA RECIBIDA" in desc_upper or "CREDITO" in desc_upper or "DEPOSITO" in desc_upper:
                es_credito = True
            elif "IMPUESTO" in desc_upper or "DEBITO" in desc_upper or "PAGO" in desc_upper or "COMISION" in desc_upper or "IV.A." in desc_upper:
                es_credito = False
            else:
                # Heurística 2: Posición en la string
                # Si el número aparece muy al final de una linea larga, puede ser Haber
                pass

            if es_credito:
                deb = 0.0; cre = parsed['valor_bruto']
            else:
                deb = parsed['valor_bruto']; cre = 0.0
            
            datos.append({
                "Fecha": parsed['fecha'],
                "Descripción": parsed['descripcion'],
                "Débito": deb,
                "Crédito": cre,
                "Origen": "OCR"
            })
            
    return pd.DataFrame(datos)


# --- UI ---
st.title("🤖 Lector Credicoop Dual (Texto + OCR)")

with st.sidebar:
    st.info("Este sistema intenta leer texto. Si falla, usa inteligencia visual (OCR).")
    x_corte = st.slider("Corte visual (Solo PDF Nativo)", 350, 550, 480)
    uploaded_file = st.file_uploader("Subir Resumen", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    st.write("Analizando tipo de archivo...")
    
    df_resultado = pd.DataFrame()
    metodo_usado = ""

    # INTENTO 1: NATIVO
    try:
        df_resultado = procesar_nativo(file_bytes, x_corte)
        metodo_usado = "Lectura Directa (Nativo)"
        if df_resultado.empty:
            raise Exception("Lectura vacía")
    except Exception as e:
        st.warning(f"Fallo lectura directa ({e}). Activando OCR (esto tarda más)...")
        # INTENTO 2: OCR
        with st.spinner("Escaneando imagen con IA..."):
            try:
                df_resultado = procesar_ocr(file_bytes)
                metodo_usado = "OCR (Escaneo Visual)"
            except Exception as e_ocr:
                st.error(f"Falló también el OCR: {e_ocr}")

    if not df_resultado.empty:
        st.success(f"Procesado con éxito usando: **{metodo_usado}**")
        
        # Totales
        c1, c2, c3 = st.columns(3)
        deb = df_resultado["Débito"].sum()
        cre = df_resultado["Crédito"].sum()
        c1.metric("Débitos", f"${deb:,.2f}")
        c2.metric("Créditos", f"${cre:,.2f}")
        c3.metric("Saldo Calc.", f"${(cre-deb):,.2f}")

        st.dataframe(df_resultado, use_container_width=True)
        
        # Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_resultado.to_excel(writer, index=False)
            
        st.download_button("Descargar Excel", buffer.getvalue(), "resumen.xlsx")
    else:
        st.error("No se pudieron extraer datos válidos con ningún método.")
