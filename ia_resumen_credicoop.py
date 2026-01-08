import io
import re
import pandas as pd
import streamlit as st
import pdfplumber
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="IA Resumen Credicoop", layout="wide")
st.title("🏦 IA Resumen Credicoop (Corregido y Conciliado)")

# --- UTILIDADES ---
def fmt_ar(n):
    if n is None or np.isnan(n): return "—"
    return f"{n:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")

def normalize_money(text):
    if not text: return 0.0
    s = str(text).replace(" ", "").replace("−", "-").replace(".", "").replace(",", ".")
    try: return float(s)
    except: return 0.0

def clasificar(desc):
    n = desc.upper()
    if "25413" in n: return "Ley 25.413" [cite: 82, 180]
    if "IVA" in n:
        if "10,5" in n: return "IVA 10,5%" [cite: 87]
        return "IVA 21%" [cite: 70]
    if "PERCEP" in n: return "Percepciones" [cite: 78, 99]
    if any(k in n for k in ["COMISION", "SERVICIO", "MANTEN", "GASTO"]): return "Gastos Bancarios (Neto)" [cite: 70, 78]
    if any(k in n for k in ["PRESTAMO", "CUOTA", "AMORT"]): return "Préstamos" [cite: 28, 70]
    return "Otros"

# --- PROCESAMIENTO ---
def parse_credicoop_estricto(pdf_bytes):
    rows = []
    saldo_anterior = 0.0
    saldo_final = 0.0
    current_row = None
    full_text = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # Detectar límites de columnas en página 1
        p1 = pdf.pages[0]
        words = p1.extract_words()
        # Coordenadas X promedio para Credicoop
        limite_deb_cre = 455 
        limite_cre_sal = 535

        deb_h = next((w for w in words if "DEBITO" in w['text'].upper()), None) [cite: 17, 158]
        cre_h = next((w for w in words if "CREDITO" in w['text'].upper()), None) [cite: 17, 158]
        if deb_h and cre_h:
            limite_deb_cre = (deb_h['x1'] + cre_h['x0']) / 2
            limite_cre_sal = cre_h['x1'] + 35

        for page in pdf.pages:
            full_text += (page.extract_text() or "") + "\n"
            p_words = page.extract_words(x_tolerance=2, y_tolerance=3)
            
            # Agrupar por líneas
            lines = {}
            for w in p_words:
                y = round(w['top'])
                if y not in lines: lines[y] = []
                lines[y].append(w)

            for y in sorted(lines.keys()):
                line_words = sorted(lines[y], key=lambda w: w['x0'])
                line_text = " ".join([w['text'] for w in line_words])
                
                # REGLA: La fecha marca el nuevo registro 
                match_fecha = re.match(r"^(\d{2}/\d{2}/\d{2,4})", line_text)
                
                if match_fecha:
                    if current_row: rows.append(current_row)
                    current_row = {
                        "fecha": match_fecha.group(1),
                        "descripcion": "",
                        "debito": 0.0,
                        "credito": 0.0,
                        "clasificacion": ""
                    }
                    words_to_proc = line_words
                else:
                    words_to_proc = line_words

                if current_row:
                    for w in words_to_proc:
                        txt = w['text']
                        x_center = (w['x0'] + w['x1']) / 2
                        
                        # Si es un monto (formato 1.234,55)
                        if re.match(r"^-?[\d\.]+,[\d]{2}$", txt):
                            val = normalize_money(txt)
                            if x_center < limite_deb_cre:
                                current_row["debito"] += val
                            elif x_center < limite_cre_sal:
                                current_row["credito"] += val
                            # Ignoramos la columna Saldo (derecha) 
                        else:
                            # Concatenar descripción (evitando repetir la fecha)
                            if txt not in current_row["fecha"]:
                                current_row["descripcion"] = (current_row["descripcion"] + " " + txt).strip()

        if current_row: rows.append(current_row)

        # Capturar Saldos y Cuadro Final
        match_ant = re.search(r"SALDO ANTERIOR\s+([\d\.,\-]+)", full_text) [cite: 17, 158]
        if match_ant: saldo_anterior = normalize_money(match_ant.group(1))
        
        match_fin = re.search(r"SALDO AL \d{2}/\d{2}/\d{2,4}\s+([\d\.,\-]+)", full_text) [cite: 79, 172]
        if match_fin: saldo_final = normalize_money(match_fin.group(1))

    # Post-procesar clasificaciones
    for r in rows:
        r["clasificacion"] = clasificar(r["descripcion"])

    return pd.DataFrame(rows), saldo_anterior, saldo_final, full_text

# --- UI APP ---
uploaded = st.file_uploader("Subí el PDF de Credicoop", type="pdf")

if uploaded:
    df, s_ant, s_fin, raw_text = parse_credicoop_estricto(uploaded.read())
    
    if not df.empty:
        # Conciliación
        t_deb = df["debito"].sum()
        t_cre = df["credito"].sum()
        s_calc = s_ant + t_cre - t_deb [cite: 158, 192]
        diff = s_calc - s_fin

        # Métricas principales
        st.subheader("📊 Conciliación")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Saldo Anterior", fmt_ar(s_ant)) [cite: 17, 158]
        c2.metric("Créditos (+)", fmt_ar(t_cre))
        c3.metric("Débitos (-)", fmt_ar(t_deb))
        c4.metric("Diferencia", fmt_ar(diff), delta_color="inverse")

        if abs(diff) < 1.0:
            st.success(f"✅ Conciliado. Saldo final calculado: {fmt_ar(s_calc)}")
        else:
            st.error(f"❌ Error de conciliación. Diferencia: {fmt_ar(diff)}")

        # Grillas
        tab1, tab2, tab3 = st.tabs(["📋 Movimientos", "💰 Gastos e IVA", "🏦 Préstamos"])
        
        with tab1:
            st.dataframe(df[["fecha", "descripcion", "debito", "credito"]], use_container_width=True)

        with tab2:
            st.write("### Resumen de Gastos (Basado en Movimientos)")
            gastos_res = df.groupby("clasificacion")[["debito"]].sum().reset_index()
            st.table(gastos_res[gastos_res["debito"] > 0])
            
            st.write("### Cuadro Oficial del Banco (Pie de página)")
            # Extraer del texto bruto el resumen oficial [cite: 84, 187]
            oficial = []
            for tag in ["LEY 25413", "RG 2408", "IVA ALICUOTA INSCRIPTO"]:
                m = re.search(f"{tag}.*?([\d\.,]+)", raw_text, re.S)
                if m: oficial.append({"Concepto": tag, "Monto": m.group(1)})
            if oficial: st.table(pd.DataFrame(oficial))

        with tab3:
            st.dataframe(df[df["clasificacion"] == "Préstamos"])

        # Descarga Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Conciliacion')
        st.download_button("📥 Descargar Excel", output.getvalue(), "resumen_bancario.xlsx")
