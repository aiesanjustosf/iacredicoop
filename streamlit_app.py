import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Conciliador AIE - Inteligente",
    page_icon="favicon.ico",
    layout="wide" # Cambié a wide para ver mejor las columnas
)

# --- CSS PERSONALIZADO (ESTÉTICA AIE) ---
st.markdown("""
    <style>
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
        border-left: 5px solid #E30613;
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
    .status-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .success { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
    .warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .danger { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES DE LIMPIEZA ---

def limpiar_numero_ar(valor):
    """Convierte '1.050,50' o '1.050,50-' a float."""
    if not valor: return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    
    val_str = str(valor).strip()
    es_negativo = False
    # Manejo de negativos al final (típico de bancos) o paréntesis
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
    if pd.isna(valor) or valor == "": return ""
    return "{:,.2f}".format(valor).replace(",", "X").replace(".", ",").replace("X", ".")

# --- LÓGICA DE CORRECCIÓN INTELIGENTE ---

def autocorregir_errores(df, saldo_inicial):
    """
    Recorre los movimientos y verifica contra la columna Saldo_PDF.
    Si detecta error, intenta corregirlo invirtiendo columnas.
    """
    if df.empty: return df, 0, 0
    
    df = df.copy()
    df['Estado'] = 'OK' # Para marcar filas corregidas
    df['Saldo_Calculado'] = 0.0
    
    saldo_acumulado = saldo_inicial
    correcciones = 0
    
    for i, row in df.iterrows():
        debito = row['Débito']
        credito = row['Crédito']
        saldo_pdf = row['Saldo_PDF']
        
        # 1. Calculamos saldo teórico actual
        saldo_teorico = saldo_acumulado + credito - debito
        
        # 2. Verificamos si coincide con el PDF (si hay dato en PDF)
        coincide = True
        if saldo_pdf != 0:
            diff = abs(saldo_teorico - saldo_pdf)
            if diff > 1.00: # Tolerancia $1
                coincide = False
        
        # 3. INTENTO DE CORRECCIÓN
        if not coincide and saldo_pdf != 0:
            # Caso A: Se leyó como Débito pero era Crédito
            # Si era Crédito, el saldo debería haber subido, no bajado.
            # Diferencia matemática: 2 * Monto
            saldo_si_fuera_credito = saldo_acumulado + debito # (asumiendo que el debito leído era en realidad credito)
            
            # Caso B: Se leyó como Crédito pero era Débito
            saldo_si_fuera_debito = saldo_acumulado - credito

            # Verificamos hipótesis A (El Débito en realidad es un Crédito mal ubicado)
            if debito > 0 and abs(saldo_si_fuera_credito - saldo_pdf) < 1.00:
                df.at[i, 'Crédito'] = debito
                df.at[i, 'Débito'] = 0.0
                df.at[i, 'Estado'] = 'CORREGIDO (Era Crédito)'
                saldo_acumulado = saldo_pdf # Sincronizamos
                correcciones += 1
                
            # Verificamos hipótesis B (El Crédito en realidad es un Débito mal ubicado)
            elif credito > 0 and abs(saldo_si_fuera_debito - saldo_pdf) < 1.00:
                df.at[i, 'Débito'] = credito
                df.at[i, 'Crédito'] = 0.0
                df.at[i, 'Estado'] = 'CORREGIDO (Era Débito)'
                saldo_acumulado = saldo_pdf
                correcciones += 1
                
            else:
                # No se pudo corregir automáticamente, marcamos error y forzamos sincronización
                df.at[i, 'Estado'] = 'ERROR DE LECTURA'
                # Forzamos el saldo del PDF para no arrastrar el error a las siguientes filas
                saldo_acumulado = saldo_pdf 
        else:
            # Todo OK, actualizamos acumulado
            if saldo_pdf != 0:
                saldo_acumulado = saldo_pdf # Usamos el del PDF como ancla firme
            else:
                saldo_acumulado = saldo_teorico
        
        df.at[i, 'Saldo_Calculado'] = saldo_acumulado

    return df, correcciones, saldo_acumulado

# --- PROCESAMIENTO PDF ---

def procesar_pdf(pdf_file, x_coords):
    movimientos = []
    saldo_anterior = 0.0
    
    # Desempaquetar sliders
    x_fecha, x_desc, x_debito, x_credito = x_coords

    with pdfplumber.open(pdf_file) as pdf:
        # 1. Buscar Saldo Anterior (Pág 1)
        if len(pdf.pages) > 0:
            p1 = pdf.pages[0].extract_text() or ""
            # Regex flexible para saldo anterior
            m = re.search(r"Saldo anterior[:\s]+([\d.,\-]+)", p1, re.IGNORECASE)
            if m: saldo_anterior = limpiar_numero_ar(m.group(1))

        # 2. Leer páginas
        for page in pdf.pages:
            # Configuración explícita de columnas
            # Agregamos una línea al final (page.width) para capturar el Saldo
            lines = [0, x_fecha, x_desc, x_debito, x_credito, page.width]
            
            settings = {
                "vertical_strategy": "explicit",
                "explicit_vertical_lines": lines,
                "horizontal_strategy": "text",
                "intersection_y_tolerance": 5
            }
            
            table = page.extract_table(settings)
            
            if table:
                for row in table:
                    # Limpieza básica de celdas None
                    row = [c.strip() if c else "" for c in row]
                    
                    # Filtro: Debe empezar con fecha DD/MM/YY
                    if len(row) >= 5 and re.match(r'\d{2}/\d{2}/\d{2}', row[0]):
                        # Ignorar líneas de totales parciales si aparecen duplicadas
                        if "SALDO AL" in row[1]: continue
                        
                        try:
                            movimientos.append({
                                "Fecha": row[0],
                                "Descripción": row[1],
                                "Débito": limpiar_numero_ar(row[2]),
                                "Crédito": limpiar_numero_ar(row[3]),
                                "Saldo_PDF": limpiar_numero_ar(row[4]) # Importante: Columna Saldo
                            })
                        except: pass

    return pd.DataFrame(movimientos), saldo_anterior

# --- INTERFAZ DE USUARIO ---

# Encabezado
c1, c2 = st.columns([1, 5])
with c1:
    if os.path.exists("logo_aie.png"):
        st.image("logo_aie.png", width=100)
    else:
        st.write("🏦")
with c2:
    st.title("Conciliador Inteligente AIE")

st.markdown("---")

col_izq, col_der = st.columns([1, 3])

with col_izq:
    st.header("Calibración Visual")
    st.info("Ajustá las columnas para la lectura inicial.")
    
    # SLIDERS
    x_fecha = st.slider("Corte Fecha", 0, 150, 60)
    x_desc = st.slider("Corte Descripción", 100, 500, 310)
    x_debito = st.slider("Corte Débito", 300, 600, 480)
    x_credito = st.slider("Corte Crédito", 500, 700, 580)
    
    uploaded_file = st.file_uploader("Subir PDF Credicoop", type="pdf")

if uploaded_file:
    # 1. Procesamiento Inicial (Lectura Cruda)
    x_coords = [x_fecha, x_desc, x_debito, x_credito]
    df_raw, saldo_ini_detectado = procesar_pdf(uploaded_file, x_coords)
    
    with col_izq:
        st.divider()
        saldo_inicial = st.number_input("Saldo Anterior", value=saldo_ini_detectado, step=1000.0)

    if df_raw.empty:
        st.warning("⚠️ No se leyeron datos. Mové los sliders a la izquierda.")
    else:
        # 2. PROCESO DE AUTOCORRECCIÓN
        # Aquí sucede la magia: Revisa subtotales y corrige
        df_corregido, num_correcciones, saldo_final_calc = autocorregir_errores(df_raw, saldo_inicial)
        
        # Totales
        t_deb = df_corregido["Débito"].sum()
        t_cre = df_corregido["Crédito"].sum()
        
        # Último saldo del PDF para comparar
        try:
            ultimo_saldo_pdf = df_corregido[df_corregido['Saldo_PDF'] != 0].iloc[-1]['Saldo_PDF']
        except:
            ultimo_saldo_pdf = 0.0

        diferencia_final = saldo_final_calc - ultimo_saldo_pdf

        # --- RESULTADOS VISUALES ---
        with col_der:
            # Tarjetas
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">Saldo Anterior</div>
                    <div class="metric-value">${saldo_inicial:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Créditos</div>
                    <div class="metric-value" style="color:green;">+${t_cre:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Débitos</div>
                    <div class="metric-value" style="color:red;">-${t_deb:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Saldo Final Calc.</div>
                    <div class="metric-value">${saldo_final_calc:,.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Estado de Autocorrección
            if num_correcciones > 0:
                st.markdown(f"""
                <div class="status-box warning">
                    ✨ SE CORRIGIERON AUTOMÁTICAMENTE {num_correcciones} MOVIMIENTOS <br>
                    <small>El sistema detectó columnas mal alineadas y las arregló usando el saldo parcial.</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Estado Final
            if abs(diferencia_final) < 1.00:
                st.markdown(f"""
                <div class="status-box success">
                    ✅ CONCILIACIÓN PERFECTA <br>
                    El saldo calculado coincide con el extracto.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-box danger">
                    ❌ DIFERENCIA DETECTADA: ${diferencia_final:,.2f} <br>
                    Revisá las filas marcadas como "ERROR" en la tabla.
                </div>
                """, unsafe_allow_html=True)

            # TABLA DE DATOS
            st.subheader("Detalle de Movimientos")
            
            # Colorear filas según estado
            def color_rows(row):
                if 'CORREGIDO' in row['Estado']:
                    return ['background-color: #fff3cd'] * len(row) # Amarillo suave
                elif 'ERROR' in row['Estado']:
                    return ['background-color: #f8d7da'] * len(row) # Rojo suave
                return [''] * len(row)

            # Formato visual para la tabla
            df_view = df_corregido.copy()
            cols_money = ['Débito', 'Crédito', 'Saldo_PDF', 'Saldo_Calculado']
            for c in cols_money:
                df_view[c] = df_view[c].apply(formatear_moneda_ar)

            st.dataframe(df_view.style.apply(color_rows, axis=1), use_container_width=True, height=500)
            
            # DESCARGA EXCEL
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_corregido.to_excel(writer, index=False, sheet_name='Conciliacion')
                workbook = writer.book
                worksheet = writer.sheets['Conciliacion']
                # Formato condicional básico en Excel si fuera necesario se agrega acá
            
            st.download_button("📥 Descargar Excel (.xlsx)", buffer.getvalue(), "conciliacion_aie.xlsx")

else:
    with col_der:
        st.info("Subí el PDF para comenzar.")
