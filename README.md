# IA Resumen Credicoop (AIE San Justo)

Aplicación Streamlit para procesar resúmenes bancarios de Banco Credicoop en PDF, con:

- Conciliación estricta: Saldo anterior + Créditos − Débitos = Saldo al dd/mm/aaaa.
- Parser híbrido:
  - PDF con texto: extracción por coordenadas (pdfplumber `page.chars`).
  - PDF CID/Type3: OCR estructurado (pypdfium2 + tesseract).
- Resumen Operativo (Módulo IVA) y Detalle de Préstamos.
- Descargas: Excel (con formatos), CSV y PDF del Resumen Operativo.
- Sin UI de filtros: las tablas se renderizan como HTML estático.
- Estética centrada similar a otras apps (max-width 900px).

## Deploy (Streamlit Community Cloud)

1. Subí este repo a GitHub.
2. En Streamlit Cloud:
   - Main file path: `streamlit_app.py`
   - Python: se toma de `runtime.txt` (3.12)
3. Asegurate de mantener:
   - `requirements.txt`
   - `packages.txt` (instala tesseract + idioma spa)

## Branding
Copiar en la raíz (opcional):
- `logo_aie.png`
- `favicon-aie.ico`
