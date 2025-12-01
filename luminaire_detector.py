"""
Sistema de Detecção de Luminárias
Identifica modelo e potência (Watts) de luminárias a partir de imagens
Sem uso de APIs ou modelos de IA externos
Autor: Douglas Ramos
Data: 2025
"""

import cv2
import numpy as np
import pytesseract
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Classe para armazenar resultado de uma detecção"""
    detection_id: int
    bbox: Optional[List[int]]
    model: str
    power_watts: Optional[float]
    confidence: float
    ocr_text: str
    explain: str


@dataclass
class ProcessingResult:
    """Classe para resultado completo do processamento"""
    image_id: str
    detections: List[Detection]
    processing_time_ms: int


class LuminaireDetector:
    """
    Sistema principal de detecção de luminárias
    Pipeline: Detecção -> OCR -> Pattern Matching -> Estimativa de Potência
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o detector
        
        Args:
            config_path: Caminho para arquivo de configuração JSON (opcional)
        """
        # Tabela de referência modelo -> potência
        self.model_power_reference = {
            'LUXA200': 24,
            'LUXA150': 18,
            'LUXA100': 12,
            'LUXB300': 36,
            'LUXB250': 30,
            'LUXB200': 24,
            'LUXC150': 18,
            'LUXC100': 12,
            'PHILIPS-T8': 18,
            'PHILIPS-LED': 24,
            'OSRAM-LED': 24,
            'GE-BASIC': 16,
            'INTRAL': 18,
            'LUMINUS': 24
        }
        
        # Variações de nome de modelos
        self.model_variations = {
            'LUX200': 'LUXA200',
            'LUXA-200': 'LUXA200',
            'LUX 200': 'LUXA200',
            'LUX150': 'LUXA150',
            'LUXA-150': 'LUXA150',
            'LUX 150': 'LUXA150',
            'PHILIPS T8': 'PHILIPS-T8',
            'PHILIPST8': 'PHILIPS-T8',
            'OSRAM LED': 'OSRAM-LED',
            'OSRAMLED': 'OSRAM-LED',
        }
        
        # Características visuais conhecidas de modelos (para classificação sem IA)
        self.visual_fingerprints = self._build_visual_fingerprints()
        
        # Carregar configuração customizada se fornecida
        if config_path:
            self._load_config(config_path)
        
        # Configurações de OCR
        self.tesseract_config = r'--oem 3 --psm 6'
        
        # Threshold de confiança mínima
        self.min_confidence = 0.4
        
        logger.info("LuminaireDetector inicializado (modo offline)")
    
    def _build_visual_fingerprints(self) -> Dict:
        """
        Constrói fingerprints visuais baseados em características geométricas
        Método determinístico sem IA
        """
        return {
            'LUXA200': {
                'aspect_ratio_range': (1.5, 2.5),
                'edge_density_range': (0.3, 0.5),
                'corners_range': (4, 8),
                'typical_colors': ['white', 'silver', 'gray']
            },
            'LUXA150': {
                'aspect_ratio_range': (1.2, 2.0),
                'edge_density_range': (0.25, 0.45),
                'corners_range': (4, 6),
                'typical_colors': ['white', 'beige']
            },
            'LUXB300': {
                'aspect_ratio_range': (2.0, 3.5),
                'edge_density_range': (0.35, 0.55),
                'corners_range': (6, 10),
                'typical_colors': ['silver', 'metallic']
            }
        }
    
    def _load_config(self, config_path: str):
        """Carrega configurações de arquivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.model_power_reference.update(config.get('model_power_reference', {}))
                self.model_variations.update(config.get('model_variations', {}))
                self.min_confidence = config.get('min_confidence', 0.4)
                logger.info(f"Configuração carregada de {config_path}")
        except Exception as e:
            logger.warning(f"Erro ao carregar configuração: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processa imagem para melhorar detecção e OCR
        
        Args:
            image: Imagem em formato numpy array (BGR)
            
        Returns:
            Imagem pré-processada
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aumentar contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Redução de ruído
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Aumentar nitidez
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def detect_objects(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta objetos (luminárias) na imagem usando detecção de contornos
        Método puramente baseado em OpenCV, sem deep learning
        
        Args:
            image: Imagem em formato numpy array
            
        Returns:
            Lista de bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Pré-processar
        processed = self.preprocess_image(image)
        
        # Detecção de bordas Canny
        edges = cv2.Canny(processed, 50, 150)
        
        # Dilatar para conectar componentes
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        height, width = image.shape[:2]
        image_area = width * height
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filtrar contornos muito pequenos ou muito grandes
            if 0.01 * image_area < area < 0.8 * image_area:
                # Verificar aspect ratio (luminárias geralmente são retangulares)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 5.0:
                    bboxes.append((x, y, x + w, y + h))
        
        # Se não encontrou nenhum objeto, usar imagem inteira
        if not bboxes:
            bboxes = [(0, 0, width, height)]
        
        # Ordenar por área (maiores primeiro)
        bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        
        return bboxes[:5]  # Limitar a 5 detecções
    
    def perform_ocr(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Executa OCR na região especificada usando Tesseract (offline)
        
        Args:
            image: Imagem completa
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Texto extraído
        """
        x1, y1, x2, y2 = bbox
        
        # Garantir coordenadas válidas
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ""
        
        # Pré-processar ROI
        processed_roi = self.preprocess_image(roi)
        
        # Aplicar binarização adaptativa
        binary = cv2.adaptiveThreshold(
            processed_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        try:
            # OCR com Tesseract (offline)
            text = pytesseract.image_to_string(binary, config=self.tesseract_config)
            text = text.strip()
            
            # Limpar texto
            text = re.sub(r'\s+', ' ', text)
            
            logger.debug(f"OCR extraiu: {text}")
            return text
        except Exception as e:
            logger.error(f"Erro no OCR: {e}")
            return ""
    
    def extract_power(self, text: str) -> Optional[float]:
        """
        Extrai potência em Watts do texto usando regex
        
        Args:
            text: Texto do OCR
            
        Returns:
            Potência em Watts ou None
        """
        # Padrões de regex para encontrar potência
        patterns = [
            r'(\d+\.?\d*)\s*kW',              # Ex: 1.5kW, 2 kW
            r'(\d+\.?\d*)\s*W(?:atts?)?',     # Ex: 24W, 18 Watts
            r'(\d+\.?\d*)\s*w',                # Ex: 24w (minúscula)
            r'(\d+)\s*[-/]\s*(\d+)\s*W',      # Ex: 18-24W (pegar primeiro)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                power = float(match.group(1))
                
                # Converter kW para W
                if 'kW' in match.group(0) or 'kw' in match.group(0).lower():
                    power *= 1000
                
                # Validar valor razoável (luminárias geralmente entre 5W e 500W)
                if 5 <= power <= 500:
                    logger.info(f"Potência extraída do OCR: {power}W")
                    return power
        
        return None
    
    def extract_model(self, text: str) -> Optional[str]:
        """
        Extrai nome do modelo do texto usando pattern matching
        
        Args:
            text: Texto do OCR
            
        Returns:
            Nome do modelo ou None
        """
        # Converter para maiúsculas para busca
        text_upper = text.upper()
        
        # Padrões para encontrar modelos
        patterns = [
            r'LUXA?\d+',          # Ex: LUXA200, LUX200
            r'LUXB?\d+',          # Ex: LUXB300
            r'LUXC?\d+',          # Ex: LUXC150
            r'PHILIPS[- ]?T\d+',  # Ex: PHILIPS-T8, PHILIPS T8
            r'PHILIPS[- ]?LED',   # Ex: PHILIPS-LED
            r'OSRAM[- ]?LED',     # Ex: OSRAM-LED
            r'GE[- ]?BASIC',      # Ex: GE-BASIC
            r'INTRAL',            # Ex: INTRAL
            r'LUMINUS',           # Ex: LUMINUS
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                model = match.group(0).replace(' ', '-')
                
                # Normalizar usando variações conhecidas
                if model in self.model_variations:
                    model = self.model_variations[model]
                
                logger.info(f"Modelo extraído do OCR: {model}")
                return model
        
        return None
    
    def calculate_visual_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Calcula características visuais da luminária para classificação sem IA
        
        Args:
            image: Imagem completa
            bbox: Bounding box da luminária
            
        Returns:
            Dicionário com features
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {}
        
        h, w = roi.shape[:2]
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Densidade de bordas
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h) if (w * h) > 0 else 0
        
        # Número de cantos detectados
        corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=20, qualityLevel=0.01, minDistance=10)
        num_corners = len(corners) if corners is not None else 0
        
        # Cor dominante (simplificado)
        if len(roi.shape) == 3:
            mean_color = np.mean(roi, axis=(0, 1))
            dominant_color = self._classify_color(mean_color)
        else:
            dominant_color = 'gray'
        
        return {
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'num_corners': num_corners,
            'dominant_color': dominant_color,
            'width': w,
            'height': h
        }
    
    def _classify_color(self, bgr_color: np.ndarray) -> str:
        """Classifica cor BGR em categoria"""
        b, g, r = bgr_color
        
        # Converter para HSV para melhor classificação
        hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv
        
        if s < 30:  # Baixa saturação = cinza/branco/preto
            if v > 200:
                return 'white'
            elif v < 50:
                return 'black'
            else:
                return 'gray'
        elif v > 150 and s < 100:
            return 'silver'
        else:
            return 'metallic'
    
    def classify_visual(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Classifica visualmente o modelo da luminária usando regras heurísticas
        Método determinístico sem IA
        
        Args:
            image: Imagem completa
            bbox: Bounding box da luminária
            
        Returns:
            (modelo, confiança)
        """
        features = self.calculate_visual_features(image, bbox)
        
        if not features:
            return "não identificado", 0.2
        
        # Pontuação para cada modelo
        scores = {}
        
        for model, fingerprint in self.visual_fingerprints.items():
            score = 0.0
            
            # Aspect ratio match
            ar_min, ar_max = fingerprint['aspect_ratio_range']
            if ar_min <= features['aspect_ratio'] <= ar_max:
                score += 0.4
            
            # Edge density match
            ed_min, ed_max = fingerprint['edge_density_range']
            if ed_min <= features['edge_density'] <= ed_max:
                score += 0.3
            
            # Corners match
            c_min, c_max = fingerprint['corners_range']
            if c_min <= features['num_corners'] <= c_max:
                score += 0.2
            
            # Color match
            if features['dominant_color'] in fingerprint['typical_colors']:
                score += 0.1
            
            scores[model] = score
        
        # Selecionar modelo com maior pontuação
        if scores:
            best_model = max(scores, key=scores.get)
            confidence = scores[best_model]
            
            if confidence >= 0.4:
                logger.info(f"Classificação visual: {best_model} (confiança: {confidence:.2f})")
                return best_model, confidence
        
        # Fallback: usar modelo mais comum como default
        default_model = list(self.model_power_reference.keys())[0]
        logger.info(f"Classificação visual falhou, usando default: {default_model}")
        return default_model, 0.3
    
    def process_detection(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        detection_id: int
    ) -> Detection:
        """
        Processa uma detecção individual
        
        Args:
            image: Imagem completa
            bbox: Bounding box da detecção
            detection_id: ID da detecção
            
        Returns:
            Objeto Detection com resultados
        """
        # Passo 1: OCR
        ocr_text = self.perform_ocr(image, bbox)
        
        # Passo 2: Extrair informações do OCR
        power_from_ocr = self.extract_power(ocr_text)
        model_from_ocr = self.extract_model(ocr_text)
        
        # Inicializar variáveis
        model = model_from_ocr
        power = power_from_ocr
        confidence = 0.95  # Alta confiança se OCR encontrou tudo
        explain = "OCR detectou informações completas na etiqueta"
        
        # Passo 3: Se OCR não encontrou modelo, classificar visualmente
        if not model:
            model, visual_confidence = self.classify_visual(image, bbox)
            confidence = visual_confidence
            explain = f"Classificação visual baseada em características geométricas (confiança: {visual_confidence:.2f})"
        
        # Passo 4: Se OCR não encontrou potência, usar tabela de referência
        if not power and model and model in self.model_power_reference:
            power = self.model_power_reference[model]
            if model_from_ocr:
                explain = "Modelo identificado por OCR; potência da tabela de referência"
            else:
                explain = f"Modelo identificado visualmente; potência da tabela de referência"
        
        # Passo 5: Verificar confiança mínima
        if confidence < self.min_confidence:
            model = "não identificado"
            power = None
            explain = "OCR ilegível e características visuais insuficientes para identificação confiável"
            confidence = max(confidence, 0.2)
        
        return Detection(
            detection_id=detection_id,
            bbox=list(bbox),
            model=model,
            power_watts=power,
            confidence=round(confidence, 2),
            ocr_text=ocr_text,
            explain=explain
        )
    
    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Processa uma imagem completa
        
        Args:
            image_path: Caminho para arquivo de imagem
            
        Returns:
            ProcessingResult com todas as detecções
        """
        start_time = time.time()
        
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        logger.info(f"Processando imagem: {image_path}")
        
        # Detectar objetos (luminárias)
        bboxes = self.detect_objects(image)
        logger.info(f"Detectados {len(bboxes)} objetos potenciais")
        
        # Processar cada detecção
        detections = []
        for i, bbox in enumerate(bboxes, start=1):
            detection = self.process_detection(image, bbox, detection_id=i)
            detections.append(detection)
        
        # Calcular tempo de processamento
        processing_time = int((time.time() - start_time) * 1000)
        
        result = ProcessingResult(
            image_id=Path(image_path).name,
            detections=detections,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Processamento concluído em {processing_time}ms")
        
        return result
    
    def visualize_results(self, image_path: str, result: ProcessingResult, output_path: str):
        """
        Desenha os resultados na imagem e salva
        
        Args:
            image_path: Caminho da imagem original
            result: Resultado do processamento
            output_path: Caminho para salvar imagem com resultados
        """
        image = cv2.imread(image_path)
        
        for det in result.detections:
            if det.bbox:
                x1, y1, x2, y2 = det.bbox
                
                # Cor baseada em confiança
                if det.confidence > 0.7:
                    color = (0, 255, 0)  # Verde
                elif det.confidence > 0.4:
                    color = (0, 165, 255)  # Laranja
                else:
                    color = (0, 0, 255)  # Vermelho
                
                # Desenhar bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Preparar texto
                if det.power_watts:
                    label = f"{det.model} - {det.power_watts}W ({int(det.confidence*100)}%)"
                else:
                    label = f"{det.model} ({int(det.confidence*100)}%)"
                
                # Calcular tamanho do texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Desenhar fundo do texto
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                
                # Desenhar texto
                cv2.putText(image, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Salvar imagem
        cv2.imwrite(output_path, image)
        logger.info(f"Imagem com resultados salva em: {output_path}")
    
    def save_json(self, result: ProcessingResult, output_path: str):
        """
        Salva resultado em formato JSON
        
        Args:
            result: Resultado do processamento
            output_path: Caminho para salvar JSON
        """
        # Converter para dicionário
        result_dict = {
            'image_id': result.image_id,
            'detections': [asdict(det) for det in result.detections],
            'processing_time_ms': result.processing_time_ms
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON salvo em: {output_path}")


def main():
    """Função principal para testar o sistema"""
    
    # Exemplo de uso
    detector = LuminaireDetector()
    
    # Processar uma imagem
    image_path = "luminaria_exemplo.jpg"
    
    try:
        # Processar
        result = detector.process_image(image_path)
        
        # Exibir resultado no console
        print("\n" + "="*60)
        print("RESULTADO DA DETECÇÃO")
        print("="*60)
        print(f"Imagem: {result.image_id}")
        print(f"Tempo: {result.processing_time_ms}ms")
        print(f"Detecções: {len(result.detections)}")
        print()
        
        for det in result.detections:
            print(f"Detecção #{det.detection_id}:")
            print(f"  Modelo: {det.model}")
            print(f"  Potência: {det.power_watts}W" if det.power_watts else "  Potência: Não identificada")
            print(f"  Confiança: {det.confidence*100:.1f}%")
            print(f"  OCR: '{det.ocr_text}'")
            print(f"  Explicação: {det.explain}")
            print()
        
        # Salvar visualização
        detector.visualize_results(image_path, result, "resultado_visualizado.jpg")
        
        # Salvar JSON
        detector.save_json(result, "resultado.json")
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()