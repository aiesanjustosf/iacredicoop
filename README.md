# IA Resumen Credicoop (AIE San Justo)

Aplicación Streamlit para procesar resúmenes bancarios de Banco Credicoop en PDF.

## Qué hace

- Extrae movimientos desde la tabla **FECHA / COMBTE / DESCRIPCION / DEBITO / CREDITO / SALDO**
- Calcula conciliación **estricta**: `Saldo anterior + Créditos − Débitos = Saldo al <fecha>`
- Toma el saldo **solo** de:
  - `SALDO ANTERIOR` (inicial)
  - `SALDO AL <dd/mm/aa|aaaa>` (final)
- Genera **Resumen Operativo** (comisiones, IVA, Ley 25.413, etc.)
- Detecta y lista **Préstamos** (si aparecen) y **Créditos no préstamo**
- Exporta **Excel** y **PDF** (Resumen Operativo)

## Lectura

1. Intenta lectura por **CHARS** (texto real del PDF).
2. Si el PDF es escaneado o no trae texto utilizable, cae a **OCR** automáticamente.

## Requisitos

- Python 3.12 (ver `runtime.txt`)
- `tesseract-ocr` y `tesseract-ocr-spa` (ver `packages.txt`)
