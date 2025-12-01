# 🔦 Sistema de Detecção de Luminárias

Sistema de visão computacional para identificação automática de modelo e potência (Watts) de luminárias a partir de imagens.

## 📋 Características

- **OCR Avançado**: Extração de texto de etiquetas e rótulos
- **Classificação Visual**: Identificação por comparação visual quando OCR falha
- **Detecção de Potência**: Extrai valores em W, kW automaticamente
- **Bounding Boxes**: Localização precisa das luminárias na imagem
- **API REST**: Endpoint para integração com outros sistemas
- **Processamento em Lote**: Múltiplas imagens simultaneamente
- **Visualização**: Imagens anotadas com detecções

## 🚀 Instalação

### 1. Requisitos do Sistema

- Python 3.8+
- Tesseract OCR
- CUDA (opcional, para GPU)

### 2. Instalar Tesseract

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-por
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Baixe o instalador em: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Instalar Dependências Python

```bash
pip install -r requirements.txt
```

## 📦 Estrutura do Projeto

```
luminaire-detector/
├── luminaire_detector.py    # Classe principal do detector
├── api.py                    # API REST com FastAPI
├── train_model.py            # Script de treinamento
├── config.json               # Configurações
├── requirements.txt          # Dependências
├── README.md                 # Esta documentação
├── dataset/                  # Dataset para treinamento
│   └── luminaires/
│       ├── LUXA200/
│       ├── LUXA150/
│       └── ...
├── uploads/                  # Imagens temporárias
└── results/                  # Resultados (JSON + visualizações)
```

## 🎯 Uso Básico

### Exemplo Simples

```python
from luminaire_detector import LuminaireDetector

# Inicializar detector
detector = LuminaireDetector(config_path="config.json")

# Processar imagem
result = detector.process_image("luminaria.jpg")

# Exibir resultados
for det in result.detections:
    print(f"Modelo: {det.model}")
    print(f"Potência: {det.power_watts}W")
    print(f"Confiança: {det.confidence*100:.1f}%")

# Salvar visualização
detector.visualize_results("luminaria.jpg", result, "resultado.jpg")

# Salvar JSON
detector.save_json(result, "resultado.json")
```

### Executar Detector via CLI

```bash
python luminaire_detector.py
```

## 🌐 API REST

### Iniciar Servidor

```bash
python api.py
```

Servidor estará disponível em: `http://localhost:8000`

### Endpoints

#### 1. Detectar Luminária (Upload de Imagem)

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@luminaria.jpg" \
  -F "save_visualization=true"
```

**Resposta:**
```json
{
  "image_id": "luminaria.jpg",
  "detections": [
    {
      "detection_id": 1,
      "bbox": [120, 80, 480, 360],
      "model": "LUXA200",
      "power_watts": 24,
      "confidence": 0.98,
      "ocr_text": "MODEL: LUXA200 24W",
      "explain": "OCR detectou informações na etiqueta"
    }
  ],
  "processing_time_ms": 312,
  "visualization_url": "/results/result_luminaria.jpg",
  "json_url": "/results/result_luminaria.json"
}
```

#### 2. Processamento em Lote

```bash
curl -X POST "http://localhost:8000/detect/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"
```

#### 3. Listar Modelos Conhecidos

```bash
curl http://localhost:8000/models
```

#### 4. Health Check

```bash
curl http://localhost:8000/health
```

### Swagger UI

Acesse a documentação interativa em: `http://localhost:8000/docs`

## 🧠 Treinamento do Modelo

### 1. Preparar Dataset

Organize imagens na seguinte estrutura:

```
dataset/luminaires/
├── LUXA200/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── LUXA150/
│   ├── img001.jpg
│   └── ...
└── LUXB300/
    └── ...
```

**Recomendações:**
- Mínimo 50-200 imagens por modelo
- Variar ângulos, iluminação e distância
- Incluir etiquetas legíveis e ilegíveis
- Usar data augmentation automática

### 2. Treinar Classificador

```python
from train_model import LuminaireClassifier

# Criar classificador
classifier = LuminaireClassifier(num_classes=10, model_name='resnet50')

# Preparar dados
train_loader, val_loader = classifier.prepare_data(
    'dataset/luminaires', 
    batch_size=16, 
    val_split=0.2
)

# Treinar
history = classifier.train(
    train_loader, val_loader,
    epochs=50,
    lr=0.001,
    save_path='luminaire_classifier.pth'
)

# Avaliar
classifier.evaluate(val_loader)
```

Ou via CLI:
```bash
python train_model.py
```

### 3. Integrar Modelo Treinado

```python
from luminaire_detector import LuminaireDetector
import torch

detector = LuminaireDetector()

# Carregar modelo treinado
checkpoint = torch.load('luminaire_classifier.pth')
detector.model.load_state_dict(checkpoint['model_state_dict'])
```

## ⚙️ Configuração

Edite `config.json` para customizar:

### Tabela de Referência Modelo → Potência

```json
{
  "model_power_reference": {
    "LUXA200": 24,
    "LUXA150": 18,
    "PHILIPS-LED-24W": 24
  }
}
```

### Parâmetros de OCR

```json
{
  "ocr_config": {
    "tesseract_mode": 3,
    "page_segmentation_mode": 6,
    "language": "por+eng"
  }
}
```

### Threshold de Confiança

```json
{
  "min_confidence": 0.4
}
```

## 📊 Formato de Saída JSON

```json
{
  "image_id": "luminaria.jpg",
  "detections": [
    {
      "detection_id": 1,
      "bbox": [x_min, y_min, x_max, y_max],
      "model": "LUXA200",
      "power_watts": 24,
      "confidence": 0.98,
      "ocr_text": "MODEL: LUXA200 24W",
      "explain": "OCR detectou informações na etiqueta"
    }
  ],
  "processing_time_ms": 312
}
```

## 🔧 Melhorias Avançadas

### 1. Usar YOLO para Detecção

```python
from ultralytics import YOLO

# Treinar YOLO custom
model = YOLO('yolov8n.pt')
model.train(data='luminaires.yaml', epochs=100)

# Integrar no detector
detector.yolo_model = YOLO('best.pt')
```

### 2. Usar CLIP para Zero-Shot Classification

```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32")

# Classificar sem treinamento
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") 
                         for c in class_names])
```

### 3. Melhorar OCR com EasyOCR

```python
import easyocr

reader = easyocr.Reader(['pt', 'en'])
result = reader.readtext(image)
```

## 📈 Métricas de Avaliação

- **Acurácia do Modelo**: % de modelos corretamente identificados
- **MAE de Potência**: Erro absoluto médio na estimativa de potência
- **Taxa de OCR**: Precision/Recall do OCR
- **Confiança Média**: % de detecções com confidence ≥ 0.8

## 🐛 Troubleshooting

### Tesseract não encontrado

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### CUDA Out of Memory

Reduza batch_size ou use CPU:
```python
detector.device = torch.device('cpu')
```

### OCR com baixa precisão

- Aumente contraste da imagem
- Use pré-processamento mais agressivo
- Ajuste parâmetros do Tesseract

## 📝 Licença

MIT License - Livre para uso comercial e pessoal

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para o branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📧 Contato

Para dúvidas ou suporte, abra uma issue no GitHub.

---

**Desenvolvido com ❤️ usando Python, OpenCV, PyTorch e Tesseract**