# IA Resumen Credicoop

App Streamlit para procesar Resumen de Cuenta Corriente Comercial (Banco Credicoop).

## Reglas implementadas
- Movimiento = línea con FECHA; líneas siguientes sin fecha se concatenan.
- Si hay 2 importes en una línea, el de la derecha es SALDO diario (se ignora).
- Débito a la izquierda; Crédito a la derecha.
- Conciliación estricta: Saldo anterior + Créditos - Débitos = Saldo AL (fecha).
- Parser híbrido: CHARS y OCR; se elige el modo que mejor concilia.

## Deploy
Subir este repo a Streamlit Cloud. Entry point: `streamlit_app.py`.
