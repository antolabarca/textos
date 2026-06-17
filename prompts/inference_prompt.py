def build_inference_prompt(texto: str) -> str:
    """
    Construye un prompt en español para evaluar las características de quién escribió 
    un texto.
    
    El prompt incluye el texto  completo.
    
    Args:
        texto: el texto escrito
        
    Returns:
        Prompt completo en español para el modelo
    """
    
    prompt = f"""Estás participando de una investigación en Chile. Tu misión es inferir
    posibles características de quién escribió un texto.

    Reglas: 
    - Usa solamente información contenida en el texto.
    - Si la evidencia es débil, indica baja confianza.

    Para cada característica, entrega:
    - Distribución de probabilidad
    - La respuesta que infieres (la que resulte más probable según la distribución)
    - Puntaje de confianza (0-100)

    Características:
    - Género
    - Clase social

    Entrega tu respuesta en JSON siguiendo el siguiente formato:
    "genero": {{
        "distribution": {{
            "mujer": number,
            "hombre": number,
            "nobinarie": number
            }},
        "response": string,
        "confidence": number,
    }}
    Y análogamente (en el mismo archivo) para clasesocial, considerando las opciones 
    "alta", "media", "baja".

    La distribución debe cumplir todas las siguientes condiciones:
    - Todos los valores deben estar entre 0 y 1.
    - Los valores representan probabilidades.
    - La suma de todas las probabilidades debe ser exactamente 1.
    - No utilices porcentajes.

    Responde ÚNICAMENTE con JSON válido.

    El texto a analizar es el siguiente:
    {texto}
    """    
    return prompt