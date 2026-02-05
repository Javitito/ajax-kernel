#!/usr/bin/env python3
"""
Adaptador de Gemini para el sistema LEANN
Integra Gemini como miembro del Consejo de Sabios con protocolos de comunicación estructurados
"""

import json
import os
import sys
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import httpx
from datetime import datetime

# Añadir el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

class GeminiContract:
    """Representa un contrato de tarea para Gemini según el Protocolo de Invocación de Tarea (PIT)"""
    
    def __init__(self, 
                 objetivo: str,
                 agente_solicitante: str = "orquestador",
                 prioridad: int = 3,
                 contexto: Optional[Dict[str, Any]] = None,
                 restricciones: Optional[Dict[str, Any]] = None,
                 presupuesto_computacional: int = 100):
        """
        Inicializa un contrato de tarea para Gemini.
        
        Args:
            objetivo: Descripción clara y concisa de la meta final
            agente_solicitante: Nombre del agente que solicita la tarea
            prioridad: Prioridad de la tarea (1-5, siendo 1 crítica)
            contexto: Contexto adicional para la tarea
            restricciones: Restricciones para la ejecución
            presupuesto_computacional: Unidades abstractas para limitar el uso
        """
        self.id_tarea = str(uuid.uuid4())
        self.agente_solicitante = agente_solicitante
        self.prioridad = prioridad
        self.objetivo = objetivo
        self.contexto = contexto or {}
        self.restricciones = restricciones or {}
        self.presupuesto_computacional = presupuesto_computacional
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el contrato a diccionario para serialización JSON"""
        return {
            "id_tarea": self.id_tarea,
            "agente_solicitante": self.agente_solicitante,
            "prioridad": self.prioridad,
            "objetivo": self.objetivo,
            "contexto": self.contexto,
            "restricciones": self.restricciones,
            "presupuesto_computacional": self.presupuesto_computacional,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiContract":
        """Crea un contrato desde diccionario"""
        contract = cls(
            objetivo=data["objetivo"],
            agente_solicitante=data.get("agente_solicitante", "orquestador"),
            prioridad=data.get("prioridad", 3),
            contexto=data.get("contexto"),
            restricciones=data.get("restricciones"),
            presupuesto_computacional=data.get("presupuesto_computacional", 100)
        )
        contract.id_tarea = data.get("id_tarea", contract.id_tarea)
        contract.timestamp = data.get("timestamp", contract.timestamp)
        return contract


class GeminiAdvisor:
    """Adaptador principal para integrar Gemini en el enjambre LEANN"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el adaptador de Gemini.
        
        Args:
            api_key: Clave API de Gemini (si no se proporciona, se carga de variables de entorno)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json"
            },
            timeout=30
        )
    
    async def execute_task(self, contract: GeminiContract) -> Dict[str, Any]:
        """
        Ejecuta una tarea mediante Gemini usando el protocolo de contrato.
        
        Args:
            contract: Contrato de tarea a ejecutar
            
        Returns:
            Dict con la respuesta de Gemini
        """
        try:
            # Preparar el prompt estructurado según el contrato
            prompt = self._build_structured_prompt(contract)
            
            # Configurar restricciones de formato de salida
            response_format = self._get_response_format(contract)
            
            # Llamar a la API de Gemini
            response = await self._call_gemini_api(prompt, response_format)
            
            return {
                "ok": True,
                "task_id": contract.id_tarea,
                "response": response,
                "agent": "gemini",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "ok": False,
                "task_id": contract.id_tarea,
                "error": str(e),
                "agent": "gemini",
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_structured_prompt(self, contract: GeminiContract) -> str:
        """Construye un prompt estructurado basado en el contrato"""
        prompt_parts = []
        
        # Encabezado con información de tarea
        prompt_parts.append(f"# TAREA GEMINI - ID: {contract.id_tarea}")
        prompt_parts.append(f"## Solicitante: {contract.agente_solicitante}")
        prompt_parts.append(f"## Prioridad: {contract.prioridad}/5")
        prompt_parts.append(f"## Presupuesto Computacional: {contract.presupuesto_computacional} unidades")
        prompt_parts.append("")
        
        # Objetivo principal
        prompt_parts.append("## OBJETIVO PRINCIPAL:")
        prompt_parts.append(contract.objetivo)
        prompt_parts.append("")
        
        # Contexto si está disponible
        if contract.contexto:
            prompt_parts.append("## CONTEXTO ADICIONAL:")
            if "descripcion" in contract.contexto:
                prompt_parts.append(f"Descripción: {contract.contexto['descripcion']}")
            if "artefactos_relevantes" in contract.contexto:
                prompt_parts.append("Artefactos relevantes:")
                for artefacto in contract.contexto["artefactos_relevantes"]:
                    prompt_parts.append(f"  - {artefacto}")
            if "memoria_corto_plazo" in contract.contexto:
                prompt_parts.append(f"Memoria reciente: {contract.contexto['memoria_corto_plazo']}")
            prompt_parts.append("")
        
        # Restricciones si están definidas
        if contract.restricciones:
            prompt_parts.append("## RESTRICCIONES Y REQUISITOS:")
            for key, value in contract.restricciones.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        # Instrucciones finales
        prompt_parts.append("## INSTRUCCIONES PARA LA RESPUESTA:")
        prompt_parts.append("Proporcione una respuesta estructurada, clara y exhaustiva.")
        prompt_parts.append("Siga el formato solicitado en las restricciones.")
        prompt_parts.append("Sea conciso pero completo.")
        prompt_parts.append("Incluya explicaciones cuando sea necesario.")
        
        return "\n".join(prompt_parts)
    
    def _get_response_format(self, contract: GeminiContract) -> str:
        """Obtiene el formato de respuesta solicitado"""
        format_requested = contract.restricciones.get("formato_salida", "text")
        
        format_descriptions = {
            "python_code": "Proporcione código Python válido con comentarios explicativos",
            "markdown_report": "Genere un informe en formato Markdown con secciones claras",
            "json_plan": "Devuelva un plan estructurado en formato JSON",
            "text": "Respuesta en texto plano estructurado"
        }
        
        return format_descriptions.get(format_requested, format_descriptions["text"])
    
    async def _call_gemini_api(self, prompt: str, response_format: str) -> str:
        """Llama a la API de Gemini con el prompt proporcionado"""
        # Este es un esquema básico - en producción se conectaría a la API real
        request_body = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "responseMimeType": "text/plain"
            }
        }
        
        # En una implementación real, aquí se haría la llamada HTTP real
        # Por ahora, simulamos una respuesta
        simulated_response = f"""Respuesta simulada para la tarea:
        
[PROMPT SIMULADO]
{prompt[:200]}...

[FIN DEL PROMPT SIMULADO]

Esta sería la respuesta real de Gemini basada en el prompt proporcionado.
El formato solicitado era: {response_format}

En una implementación completa, esta función haría una llamada real a la API de Gemini."""
        
        return simulated_response


# Funciones de utilidad para integración con el sistema
async def create_and_execute_gemini_task(
    objetivo: str,
    contexto: Optional[Dict[str, Any]] = None,
    formato_salida: str = "text",
    prioridad: int = 3
) -> Dict[str, Any]:
    """
    Crea y ejecuta una tarea de Gemini de forma sencilla.
    
    Args:
        objetivo: Objetivo de la tarea
        contexto: Contexto adicional
        formato_salida: Formato de salida solicitado
        prioridad: Prioridad de la tarea (1-5)
        
    Returns:
        Dict con el resultado de la tarea
    """
    try:
        # Crear contrato de tarea
        contract = GeminiContract(
            objetivo=objetivo,
            contexto=contexto,
            prioridad=prioridad,
            restricciones={"formato_salida": formato_salida}
        )
        
        # Crear adaptador y ejecutar tarea
        advisor = GeminiAdvisor()
        result = await advisor.execute_task(contract)
        
        return result
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "agent": "gemini",
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Función principal para uso desde línea de comandos"""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description='Adaptador de Gemini para LEANN')
    parser.add_argument('--objetivo', '-o', type=str, required=True, 
                       help='Objetivo de la tarea')
    parser.add_argument('--contexto', '-c', type=str, 
                       help='Contexto en formato JSON')
    parser.add_argument('--formato', '-f', type=str, default='text',
                       choices=['text', 'python_code', 'markdown_report', 'json_plan'],
                       help='Formato de salida')
    parser.add_argument('--prioridad', '-p', type=int, default=3,
                       choices=[1, 2, 3, 4, 5],
                       help='Prioridad de la tarea (1-5)')
    
    args = parser.parse_args()
    
    # Parsear contexto si se proporciona
    contexto = None
    if args.contexto:
        try:
            contexto = json.loads(args.contexto)
        except json.JSONDecodeError:
            print("Error: Contexto debe ser un JSON válido")
            sys.exit(1)
    
    # Ejecutar tarea
    async def run_task():
        result = await create_and_execute_gemini_task(
            objetivo=args.objetivo,
            contexto=contexto,
            formato_salida=args.formato,
            prioridad=args.prioridad
        )
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    asyncio.run(run_task())


if __name__ == "__main__":
    main()