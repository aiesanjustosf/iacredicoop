import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO
import operator

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Conciliador Automático Credicoop", layout="wide")

# --- FUNCIONES DE LIMPIEZA ---

def limpiar_numero_ar(valor):
    """Convierte string formato '1.000,00' a float de forma robusta."""
    if not valor:
        return 0.0
    
    # Si ya es número, devolverlo
    if isinstance(valor, (int, float)):
        return float(valor)
    
    val_str = str(valor).strip()
    
    # Detectar negativos (formato "1.000,00-" o "(1.000,00)")
    es_negativo = False
    if val_str.endswith("-") or (val_str.startswith("(") and val_str.endswith(")")):
        es_negativo = True
    
    # Limpiar caracteres no numéricos excepto , y .
    val_str = re.sub(r'[^\d,.]', '', val_str)
    
    if not val_str:
        return 0.0
        
    try:
        # Formato AR: 1.000,00 -> 1000.00
        val_str = val_str.replace(".", "").replace(",", ".")
        numero = float(val_str)
        return -numero if es_negativo else numero
    except ValueError:
        return 0.0

def formatear_moneda_ar(valor):
    if pd.isna(valor) or valor == "":
        return ""
    # Formato visual AR
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- LÓGICA DE EXTRACCIÓN INTELIGENTE (SIN SLIDERS) ---

def es_fecha(texto):
    # Detecta DD/MM/YY o DD/MM/AAAA
    return bool(re.match(r'^\d{2}/\d{2}/\d{2,4}$', texto))

def es_numero_moneda(texto):
    # Detecta si parece un monto (tiene dígitos y coma o punto)
    return bool(re.search(r'\d+[.,]\d+', texto))

def procesar_pdf_inteligente(pdf_file):
    movimientos = []
    saldo_anterior = 0.0
    
    # Puntos de corte APROXIMADOS para Credicoop (Hoja A4 estándar)
    # X0 es la coordenada izquierda de la palabra.
    # Zona Débito: aprox 380 a 510 puntos
    # Zona Crédito: aprox 510 a 620 puntos
    # Zona Saldo: > 620 puntos
    LIMIT_DEBITO_START = 350
    LIMIT_CREDITO_START = 510
    LIMIT_SALDO_START = 620
    
    with pdfplumber.open(pdf_file) as pdf:
        # 1. Buscar Saldo Anterior en pág 1
        if len(pdf.pages) > 0:
            text_p1 = pdf.pages[0].extract_text() or ""
            match = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", text_p1, re.IGNORECASE)
            if match:
                saldo_anterior = limpiar_numero_ar(match.group(1))

        # 2. Procesar páginas geométricamente
        for page in pdf.pages:
            # Extraemos TODAS las palabras con sus coordenadas (x, y)
            words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
            
            # Agrupar palabras por renglón (usando su posición 'top' con pequeña tolerancia)
            # Esto reconstruye las líneas visuales
            rows = {}
            for w in words:
                # Redondeamos 'top' para agrupar palabras en la misma línea visual
                row_key = round(w['top'] / 5) * 5 
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append(w)
            
            # Ordenar renglones de arriba a abajo
            sorted_row_keys = sorted(rows.keys())
            
            for key in sorted_row_keys:
                line_words = rows[key]
                # Ordenar palabras de izquierda a derecha en el renglón
                line_words.sort(key=operator.itemgetter('x0'))
                
                # VALIDACIÓN 1: ¿Empieza con fecha?
                first_word = line_words[0]['text']
                if not es_fecha(first_word):
                    continue # No es movimiento, saltar
                
                fecha = first_word
                
                # Variables para esta fila
                descripcion_parts = []
                debito = 0.0
                credito = 0.0
                saldo_linea = 0.0
                
                # Iterar el resto de palabras para clasificar
                # Empezamos desde la segunda palabra
                for w in line_words[1:]:
                    text = w['text']
                    x_pos = w['x0'] # Posición horizontal inicial
                    
                    if es_numero_moneda(text):
                        valor = limpiar_numero_ar(text)
                        
                        # CLASIFICACIÓN GEOMÉTRICA AUTOMÁTICA
                        if x_pos > LIMIT_SALDO_START:
                            saldo_linea = valor
                        elif x_pos > LIMIT_CREDITO_START:
                            credito = valor
                        elif x_pos > LIMIT_DEBITO_START:
                            debito = valor
                        else:
                            # Si es un número pero está muy a la izquierda, es parte de la descripción
                            # (Ej: "Cuota 12 de 12")
                            descripcion_parts.append(text)
                    else:
                        # Si no parece número, es descripción
                        descripcion_parts.append(text)
                
                descripcion = " ".join(descripcion_parts)
                
                # Guardar movimiento
                movimientos.append({
                    "Fecha": fecha,
                    "Descripcion": descripcion,
                    "Debito": debito,
                    "Credito": credito,
                    "Saldo_PDF": saldo_linea
                })

    return pd.DataFrame(movimientos), saldo_anterior

# --- LÓGICA DE CONTROL ---

def verificar_conciliacion(df, saldo_inicial):
    if df.empty:
        return df, 0, 0, 0
        
    df['Saldo_Calculado'] = 0.0
    df['Estado'] = 'OK'
    df['Diferencia'] = 0.0
    
    saldo_acum = saldo_inicial
    total_cred = df['Credito'].sum()
    total_deb = df['Debito'].sum()
    
    for idx, row in df.iterrows():
        saldo_acum += (row['Credito'] - row['Debito'])
        df.at[idx, 'Saldo_Calculado'] = saldo_acum
        
        # Validar Checkpoint
        saldo_pdf = row['Saldo_PDF']
        if saldo_pdf != 0:
            diff = round(saldo_acum - saldo_pdf, 2)
            if abs(diff) > 1.00:
                df.at[idx, 'Estado'] = 'ERROR'
                df.at[idx, 'Diferencia'] = diff
            else:
                # Sincronizar para evitar arrastre de decimales
                saldo_acum = saldo_pdf
                
    return df, total_cred, total_deb, saldo_acum

# --- FRONTEND (UI) ---

st.title("⚡ Conciliador Credicoop Automático")
st.markdown("Sin configuraciones manuales. Subí el PDF y el sistema detectará las columnas automáticamente.")

uploaded_file = st.file_uploader("Arrastrá tu extracto PDF aquí", type="pdf")

if uploaded_file:
    try:
        # 1. Procesar (Magia Automática)
        with st.spinner('Analizando documento con IA...'):
            df_raw, saldo_ini_auto = procesar_pdf_inteligente(uploaded_file)
        
        # 2. Input Saldo (Editable pero pre-llenado)
        col1, col2 = st.columns([1, 3])
        with col1:
            saldo_inicial = st.number_input("Saldo Anterior", value=saldo_ini_auto, step=1000.0)
        
        if df_raw.empty:
            st.error("❌ No se encontraron movimientos. Asegurate de que sea un PDF de Credicoop original (no escaneado como imagen).")
        else:
            # 3. Calcular Conciliación
            df_fin, t_cred, t_deb, saldo_fin = verificar_conciliacion(df_raw, saldo_inicial)
            
            # 4. Métricas
            with col2:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Créditos", formatear_moneda_ar(t_cred))
                c2.metric("Total Débitos", formatear_moneda_ar(t_deb))
                c3.metric("Saldo Final Calculado", formatear_moneda_ar(saldo_fin))

            st.divider()

            # 5. Verificación de Errores
            errores = df_fin[df_fin['Estado'] == 'ERROR']
            
            if not errores.empty:
                st.error(f"⚠️ Atención: Hay {len(errores)} diferencias detectadas.")
                st.write("El sistema detectó que el saldo calculado no coincide con el impreso en estas líneas:")
                
                # Tabla de errores formateada
                err_view = errores[['Fecha', 'Descripcion', 'Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']].copy()
                for c in ['Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']:
                    err_view[c] = err_view[c].apply(formatear_moneda_ar)
                st.dataframe(err_view, use_container_width=True)
            else:
                st.success("✅ Conciliación Perfecta. Todos los números cierran.")

            # 6. Tabla y Exportación
            with st.expander("Ver Detalle Completo", expanded=True):
                # Formateo para vista
                df_view = df_fin.copy()
                for c in ['Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia']:
                    df_view[c] = df_view[c].apply(formatear_moneda_ar)
                st.dataframe(df_view, use_container_width=True, height=500)
            
            # Botón Excel
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_fin.to_excel(writer, index=False, sheet_name='Conciliacion')
            
            st.download_button(
                "📥 Descargar Excel", 
                data=buffer.getvalue(), 
                file_name="conciliacion_automatica.xlsx", 
                mime="application/vnd.ms-excel",
                type="primary"
            )

    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
