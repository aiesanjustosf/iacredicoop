import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Conciliador Credicoop Pro", layout="wide")

# --- FUNCIONES DE FORMATO Y LIMPIEZA ---

def limpiar_numero_ar(valor):
    """Convierte string formato '1.000,00' a float. Devuelve 0.0 si está vacío o falla."""
    if not valor:
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    
    # Limpieza básica de basura OCR o espacios
    valor = str(valor).replace(" ", "")
    
    # Manejo específico para negativos con signo al final (común en bancos) o entre paréntesis
    es_negativo = False
    if valor.endswith("-") or (valor.startswith("(") and valor.endswith(")")):
        es_negativo = True
    
    # Quitar caracteres no numéricos excepto coma y punto
    valor = re.sub(r'[^\d,.]', '', valor)
    
    if not valor:
        return 0.0
        
    try:
        # Lógica AR: Eliminar puntos de miles, reemplazar coma decimal por punto
        valor = valor.replace(".", "").replace(",", ".")
        numero = float(valor)
        return -numero if es_negativo else numero
    except ValueError:
        return 0.0

def formatear_moneda_ar(valor):
    """Convierte float a string formato '1.000,00'"""
    if pd.isna(valor):
        return ""
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- LÓGICA DE EXTRACCIÓN (CORE) ---

def procesar_pdf(pdf_file, x_coords):
    """
    Extrae datos usando líneas verticales explícitas (x_coords).
    x_coords debe ser una lista: [fin_fecha, fin_desc, fin_debito, fin_credito]
    """
    data = []
    saldo_anterior = 0.0
    
    with pdfplumber.open(pdf_file) as pdf:
        # Intentamos buscar el saldo anterior en la primera página
        text_page1 = pdf.pages[0].extract_text()
        # Regex simple para buscar "Saldo Anterior" (ajustar según formato real si es necesario)
        match_saldo = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", text_page1, re.IGNORECASE)
        if match_saldo:
            saldo_anterior = limpiar_numero_ar(match_saldo.group(1))

        for page in pdf.pages:
            # Definimos la configuración de la tabla basada en los sliders
            # Credicoop estructura: Fecha | Descripción | Débito | Crédito | Saldo
            # Los sliders definen las líneas divisorias verticales
            table_settings = {
                "vertical_strategy": "explicit",
                "explicit_vertical_lines": [0] + x_coords + [page.width],
                "horizontal_strategy": "text", # Usa la posición del texto para definir filas
                "intersection_y_tolerance": 5, 
            }
            
            table = page.extract_table(table_settings)
            
            if table:
                for row in table:
                    # Limpieza de None
                    row = [cell.strip() if cell else "" for cell in row]
                    
                    # Validar si es una fila de movimiento (debe tener fecha válida al inicio)
                    # Formato fecha aprox: DD/MM/YY
                    if re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                        try:
                            fecha = row[0]
                            desc = row[1] # Descripción
                            
                            # A veces las columnas se desplazan si el PDF es complejo, 
                            # pero con líneas explícitas es difícil que pase.
                            debito = limpiar_numero_ar(row[2])
                            credito = limpiar_numero_ar(row[3])
                            saldo_parcial = limpiar_numero_ar(row[4]) # Columna Saldo (Control)
                            
                            data.append({
                                "Fecha": fecha,
                                "Descripcion": desc,
                                "Debito": debito,
                                "Credito": credito,
                                "Saldo_PDF": saldo_parcial
                            })
                        except Exception as e:
                            pass # Saltar filas que no sean datos puros

    return pd.DataFrame(data), saldo_anterior

# --- LÓGICA DE CONCILIACIÓN ---

def verificar_integridad(df, saldo_inicial):
    if df.empty:
        return df, 0, 0, 0

    df['Saldo_Calculado'] = 0.0
    df['Diferencia_Control'] = 0.0
    df['Estado'] = 'OK'
    
    saldo_acum = saldo_inicial
    
    # Métricas totales
    total_creditos = df['Credito'].sum()
    total_debitos = df['Debito'].sum()
    
    for index, row in df.iterrows():
        # Cálculo lógico
        saldo_acum += (row['Credito'] - row['Debito'])
        
        # Guardar en DF
        df.at[index, 'Saldo_Calculado'] = saldo_acum
        
        # Control contra la columna Saldo del PDF (si existe dato)
        saldo_pdf = row['Saldo_PDF']
        
        if saldo_pdf != 0:
            diff = round(saldo_acum - saldo_pdf, 2)
            # Tolerancia de $1.00
            if abs(diff) > 1.00:
                df.at[index, 'Diferencia_Control'] = diff
                df.at[index, 'Estado'] = 'ERROR'
                # Opcional: Auto-corregir el acumulado para no arrastrar el error
                # saldo_acum = saldo_pdf 
            else:
                # Sincronización fina (para evitar errores de punto flotante)
                saldo_acum = saldo_pdf

    saldo_final_calculado = saldo_acum
    return df, total_creditos, total_debitos, saldo_final_calculado

# --- INTERFAZ STREAMLIT ---

st.title("🏦 Conciliación Automática Credicoop v2.0")

col_config, col_main = st.columns([1, 3])

with col_config:
    st.header("Calibración")
    st.info("Ajustá las líneas verticales para que caigan ENTRE las columnas del PDF.")
    
    # Sliders para definir coordenadas X (ajustados a valores típicos de hoja A4 apaisada/vertical)
    # Asumiendo ancho aprox de 600-800 puntos
    x_fecha = st.slider("Fin Columna Fecha", 0, 200, 60, help="Donde termina la fecha y empieza la descripción")
    x_desc = st.slider("Fin Columna Descripción", 100, 600, 350, help="Donde termina descripción y empieza Débito")
    x_debito = st.slider("Fin Columna Débito", 300, 700, 480, help="Donde termina Débito y empieza Crédito")
    x_credito = st.slider("Fin Columna Crédito", 400, 800, 580, help="Donde termina Crédito y empieza Saldo")
    
    uploaded_file = st.file_uploader("Subir Extracto (PDF)", type="pdf")

if uploaded_file:
    # 1. Procesamiento
    try:
        x_coords = [x_fecha, x_desc, x_debito, x_credito]
        df_raw, saldo_inicial = procesar_pdf(uploaded_file, x_coords)
        
        # Input manual de saldo anterior por si el regex falla
        with col_config:
            saldo_inicial = st.number_input("Saldo Anterior (Manual si es necesario)", value=saldo_inicial, step=1000.0)

        # 2. Conciliación
        df_proc, t_cred, t_deb, saldo_final = verificar_integridad(df_raw, saldo_inicial)
        
        # 3. Mostrar Métricas
        with col_main:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Saldo Anterior", formatear_moneda_ar(saldo_inicial))
            m2.metric("Total Créditos", formatear_moneda_ar(t_cred), delta_color="normal")
            m3.metric("Total Débitos", formatear_moneda_ar(t_deb), delta_color="inverse")
            m4.metric("Saldo Final Calc.", formatear_moneda_ar(saldo_final))
            
            # 4. Alertas de Integridad
            errores = df_proc[df_proc['Estado'] == 'ERROR']
            if not errores.empty:
                st.error(f"⚠️ ¡Atención! Se detectaron {len(errores)} inconsistencias de lectura.")
                st.write("Estos movimientos causan diferencias entre el cálculo y lo que dice el banco:")
                
                # Tabla de errores bonita
                df_show_err = errores.copy()
                for c in ['Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia_Control']:
                    df_show_err[c] = df_show_err[c].apply(formatear_moneda_ar)
                
                st.dataframe(df_show_err[['Fecha', 'Descripcion', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia_Control']], use_container_width=True)
            else:
                st.success("✅ La conciliación matemática es perfecta. Todos los saldos parciales coinciden.")

            # 5. Tabla Principal
            st.subheader("Movimientos Detallados")
            
            # Preparar para display y excel
            df_export = df_proc.copy()
            # Formateo visual
            columnas_moneda = ['Debito', 'Credito', 'Saldo_PDF', 'Saldo_Calculado', 'Diferencia_Control']
            for col in columnas_moneda:
                df_export[col] = df_export[col].apply(formatear_moneda_ar)
                
            st.dataframe(df_export, use_container_width=True, height=500)
            
            # 6. Botón Exportar Excel
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_proc.to_excel(writer, index=False, sheet_name='Conciliacion')
                # Aquí se podría agregar formato al excel si se quisiera
                
            st.download_button(
                label="📥 Descargar Excel (.xlsx)",
                data=buffer.getvalue(),
                file_name="conciliacion_credicoop.xlsx",
                mime="application/vnd.ms-excel"
            )

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
        st.info("Prueba ajustando los sliders de columnas a la izquierda.")

else:
    with col_main:
        st.info("👆 Subí un archivo PDF para comenzar.")
