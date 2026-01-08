import io
import re
import pandas as pd
import streamlit as st
import pdfplumber
import xlsxwriter

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Conciliador Credicoop AIE", layout="wide")
st.title("🏦 Conciliador Credicoop - Procesamiento por Registros")

def limpiar_monto(texto):
    if not texto: return 0.0
    aux = re.sub(r'[^\d,\-]', '', str(texto))
    if not aux: return 0.0
    aux = aux.replace('.', '').replace(',', '.')
    try: return float(aux)
    except: return 0.0

# --- PROCESADOR ---
def procesar_banco(file_bytes):
    movimientos = []
    current_mov = None
    saldo_inicial = 0.0
    saldo_final = 0.0
    texto_total = ""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        # 1. Definir coordenadas de columnas (Basado en Página 1)
        p1 = pdf.pages[0]
        words = p1.extract_words()
        # Valores de seguridad por si no detecta encabezados
        limite_deb_cre = 460 
        limite_cre_sal = 540

        deb_head = next((w for w in words if "DEBITO" in w['text'].upper()), None)
        cre_head = next((w for w in words if "CREDITO" in w['text'].upper()), None)
        if deb_head and cre_head:
            limite_deb_cre = (deb_head['x1'] + cre_head['x0']) / 2
            limite_cre_sal = cre_head['x1'] + 30

        # 2. Procesar todas las páginas
        for page in pdf.pages:
            texto_total += (page.extract_text() or "") + "\n"
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            
            # Agrupar por líneas
            lines = {}
            for w in words:
                y = round(w['top'])
                if y not in lines: lines[y] = []
                lines[y].append(w)

            for y in sorted(lines.keys()):
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words])
                
                # Check Fecha (dd/mm/yy)
                match_fecha = re.match(r"^(\d{2}/\d{2}/\d{2,4})", line_text)
                
                if match_fecha:
                    # Guardar el movimiento anterior si existe
                    if current_mov: movimientos.append(current_mov)
                    
                    # Iniciar nuevo movimiento
                    current_mov = {
                        "Fecha": match_fecha.group(1),
                        "Descripción": "",
                        "Débito": 0.0,
                        "Crédito": 0.0
                    }
                    rest_of_words = line_words
                else:
                    rest_of_words = line_words

                if current_mov:
                    for w in rest_of_words:
                        txt = w['text']
                        x_mid = (w['x0'] + w['x1']) / 2
                        
                        # Es un monto?
                        if re.match(r"^-?[\d\.]+,[\d]{2}$", txt):
                            val = limpiar_monto(txt)
                            if x_mid < limite_deb_cre:
                                current_mov["Débito"] += val
                            elif x_mid < limite_cre_sal:
                                current_mov["Crédito"] += val
                        else:
                            # Es texto de descripción
                            if txt not in current_mov["Fecha"]:
                                current_mov["Descripción"] += " " + txt

                # Captura de Saldo Anterior
                if "SALDO ANTERIOR" in line_text.upper():
                    m = re.search(r"([\d\.]+,[\d]{2})", line_text)
                    if m: saldo_inicial = limpiar_monto(m.group(1))

        if current_mov: movimientos.append(current_mov)

        # 3. Saldo Final y Cuadro de Impuestos (Regex al texto total)
        m_fin = re.search(r"SALDO AL \d{2}/\d{2}/\d{2,4}\s+([\d\.,\-]+)", texto_total)
        saldo_final = limpiar_monto(m_fin.group(1)) if m_fin else 0.0
        
        # Extraer totales del cuadro del banco
        impuestos = []
        tags = [("TOTAL IMPUESTO LEY 25413", "Ley 25.413"), ("IVA ALIC ADIC RG 2408", "Percep. IVA"), ("IVA ALICUOTA INSCRIPTO", "IVA 21%")]
        for tag, nombre in tags:
            match = re.search(f"{tag}.*?([\d\.,]+)", texto_total, re.S)
            if match: impuestos.append({"Concepto": nombre, "Importe": limpiar_monto(match.group(1))})

    return pd.DataFrame(movimientos), saldo_inicial, saldo_final, impuestos

# --- UI ---
file = st.file_uploader("Subí el resumen", type="pdf")

if file:
    df, s_ini, s_fin, imp_oficial = procesar_banco(file.read())
    
    # Cálculos
    t_deb = df["Débito"].sum()
    t_cre = df["Crédito"].sum()
    calc = s_ini + t_cre - t_deb
    
    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saldo Anterior", f"$ {s_ini:,.2f}")
    c2.metric("Créditos (+)", f"$ {t_cre:,.2f}")
    c3.metric("Débitos (-)", f"$ {t_deb:,.2f}")
    c4.metric("Diferencia", f"$ {calc - s_fin:,.2f}")

    # Tabs
    t1, t2, t3 = st.tabs(["Movimientos", "Gastos/Impuestos", "Préstamos"])
    with t1:
        st.dataframe(df, use_container_width=True)
    with t2:
        if imp_oficial: st.table(pd.DataFrame(imp_oficial))
        else: st.info("No se detectó el cuadro resumen al final.")
    with t3:
        st.dataframe(df[df["Descripción"].str.contains("PREST|CUOTA|AMORT", case=False)])

    # Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Conciliacion')
    st.download_button("Descargar Excel", output.getvalue(), "resumen.xlsx")
