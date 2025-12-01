"""
API REST para Sistema de Detecção de Luminárias - 100% Offline
Endpoint local para upload de imagens e processamento
Sem uso de APIs externas ou modelos de IA
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path
from typing import List
import shutil

# Importar o detector
from luminaire_detector import LuminaireDetector

# Inicializar FastAPI
app = FastAPI(
    title="Luminaire Detection API (Offline)",
    description="API local para detecção de modelo e potência de luminárias - 100% Offline",
    version="2.0.0"
)

# Configurar CORS para acesso local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar detector
detector = LuminaireDetector(config_path="config.json")

# Diretórios
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Servir arquivos estáticos
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "name": "Luminaire Detection API (Offline)",
        "version": "2.0.0",
        "status": "online",
        "mode": " offline - sem APIs externas",
        "methods": [
            "OCR com Tesseract",
            "Pattern matching",
            "Análise geométrica",
            "Regras heurísticas"
        ],
        "endpoints": {
            "detect": "/detect",
            "detect_batch": "/detect/batch",
            "health": "/health",
            "models": "/models",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica saúde da API"""
    return {
        "status": "healthy",
        "detector_ready": True,
        "ocr_available": True,
        "models_loaded": len(detector.model_power_reference),
        "mode": "offline"
    }


@app.get("/models")
async def list_models():
    """Lista todos os modelos conhecidos e suas potências"""
    return {
        "total_models": len(detector.model_power_reference),
        "models": detector.model_power_reference,
        "variations": detector.model_variations
    }


@app.get("/stats")
async def get_stats():
    """Retorna estatísticas de uso"""
    results = list(RESULTS_DIR.glob("*.json"))
    images = list(RESULTS_DIR.glob("*.jpg")) + list(RESULTS_DIR.glob("*.png"))
    
    return {
        "total_processado": len(results),
        "total_visualizacoes": len(images),
        "espaco_usado_mb": sum(f.stat().st_size for f in results + images) / (1024*1024)
    }


@app.post("/detect")
async def detect_luminaire(
    file: UploadFile = File(...),
    save_visualization: bool = True
):
    """
    Detecta modelo e potência de luminária em uma imagem (offline)
    
    Args:
        file: Arquivo de imagem (JPG, PNG, etc.)
        save_visualization: Se deve salvar imagem com visualização
        
    Returns:
        JSON com detecções
    """
    # Validar tipo de arquivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    
    # Salvar arquivo temporariamente
    temp_path = UPLOAD_DIR / file.filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Processar imagem (100% offline)
        result = detector.process_image(str(temp_path))
        
        # Salvar visualização se solicitado
        visualization_path = None
        if save_visualization:
            visualization_path = RESULTS_DIR / f"result_{file.filename}"
            detector.visualize_results(str(temp_path), result, str(visualization_path))
        
        # Salvar JSON
        json_path = RESULTS_DIR / f"result_{Path(file.filename).stem}.json"
        detector.save_json(result, str(json_path))
        
        # Preparar resposta
        response = {
            "image_id": result.image_id,
            "detections": [
                {
                    "detection_id": det.detection_id,
                    "bbox": det.bbox,
                    "model": det.model,
                    "power_watts": det.power_watts,
                    "confidence": det.confidence,
                    "ocr_text": det.ocr_text,
                    "explain": det.explain
                }
                for det in result.detections
            ],
            "processing_time_ms": result.processing_time_ms,
            "visualization_url": f"/results/{visualization_path.name}" if visualization_path else None,
            "json_url": f"/results/{json_path.name}",
            "processing_mode": "offline"
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")
    
    finally:
        # Limpar arquivo temporário
        if temp_path.exists():
            temp_path.unlink()


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    save_visualizations: bool = True
):
    """
    Processa múltiplas imagens em lote (offline)
    
    Args:
        files: Lista de arquivos de imagem
        save_visualizations: Se deve salvar visualizações
        
    Returns:
        Lista de resultados JSON
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Máximo de 20 imagens por lote")
    
    results = []
    
    for file in files:
        temp_path = UPLOAD_DIR / file.filename
        
        try:
            # Salvar temporariamente
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Processar (offline)
            result = detector.process_image(str(temp_path))
            
            # Visualização
            if save_visualizations:
                visualization_path = RESULTS_DIR / f"batch_result_{file.filename}"
                detector.visualize_results(str(temp_path), result, str(visualization_path))
            
            results.append({
                "filename": file.filename,
                "success": True,
                "image_id": result.image_id,
                "detections": [
                    {
                        "detection_id": det.detection_id,
                        "model": det.model,
                        "power_watts": det.power_watts,
                        "confidence": det.confidence
                    }
                    for det in result.detections
                ],
                "processing_time_ms": result.processing_time_ms
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    return JSONResponse(content={
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", False)),
        "results": results
    })


@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Retorna arquivo de resultado (imagem ou JSON)"""
    file_path = RESULTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    return FileResponse(file_path)


@app.delete("/results/{filename}")
async def delete_result(filename: str):
    """Deleta arquivo de resultado"""
    file_path = RESULTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    
    file_path.unlink()
    return {"message": f"Arquivo {filename} deletado com sucesso"}


@app.delete("/results")
async def clear_all_results():
    """Limpa todos os resultados salvos"""
    deleted_count = 0
    
    for file_path in RESULTS_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()
            deleted_count += 1
    
    return {
        "message": f"{deleted_count} arquivo(s) deletado(s)",
        "deleted_count": deleted_count
    }


@app.post("/update-model-reference")
async def update_model_reference(model: str, power_watts: float):
    """
    Adiciona ou atualiza modelo na tabela de referência
    
    Args:
        model: Nome do modelo
        power_watts: Potência em Watts
    """
    detector.model_power_reference[model.upper()] = power_watts
    
    # Salvar em config.json
    try:
        config_path = Path("config.json")
        if config_path.exists():
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config['model_power_reference'][model.upper()] = power_watts
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass
    
    return {
        "message": f"Modelo {model} atualizado com potência {power_watts}W",
        "total_models": len(detector.model_power_reference)
    }


@app.get("/config")
async def get_config():
    """Retorna configuração atual do detector"""
    return {
        "models": detector.model_power_reference,
        "variations": detector.model_variations,
        "min_confidence": detector.min_confidence,
        "visual_fingerprints": detector.visual_fingerprints
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LUMINAIRE DETECTION API - 100% OFFLINE")
    print("="*60)
    print("Servidor iniciando...")
    print("Acesse: http://localhost:8000")
    print("Documentação: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )