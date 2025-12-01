"""
Script para treinar modelo de classificação de luminárias
Usa transfer learning com modelos pré-treinados
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class LuminaireDataset(Dataset):
    """Dataset para luminárias"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LuminaireClassifier:
    """Classificador de luminárias usando ResNet"""
    
    def __init__(self, num_classes: int, model_name: str = 'resnet50'):
        """
        Args:
            num_classes: Número de classes (modelos de luminárias)
            model_name: Nome do modelo base ('resnet50', 'efficientnet_b0', etc.)
        """
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Criar modelo
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Modelo {model_name} não suportado")
        
        self.model = self.model.to(self.device)
        print(f"Modelo criado: {model_name} com {num_classes} classes")
        print(f"Device: {self.device}")
    
    def prepare_data(
        self, 
        data_dir: str, 
        batch_size: int = 16, 
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepara dataloaders para treinamento
        
        Args:
            data_dir: Diretório com estrutura: data_dir/classe/imagem.jpg
            batch_size: Tamanho do batch
            val_split: Proporção para validação
            
        Returns:
            (train_loader, val_loader)
        """
        data_path = Path(data_dir)
        
        # Transformações
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Coletar imagens e labels
        image_paths = []
        labels = []
        class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        print(f"Classes encontradas: {class_names}")
        
        for class_name in class_names:
            class_dir = data_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(class_idx)
            
            for img_path in class_dir.glob('*.png'):
                image_paths.append(str(img_path))
                labels.append(class_idx)
        
        print(f"Total de imagens: {len(image_paths)}")
        
        # Split train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=val_split, stratify=labels, random_state=42
        )
        
        # Criar datasets
        train_dataset = LuminaireDataset(train_paths, train_labels, train_transform)
        val_dataset = LuminaireDataset(val_paths, val_labels, val_transform)
        
        # Criar dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        save_path: str = 'best_model.pth'
    ):
        """
        Treina o modelo
        
        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            epochs: Número de épocas
            lr: Learning rate
            save_path: Caminho para salvar melhor modelo
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Treino
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc='Training')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validação
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc='Validation'):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Salvar histórico
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Salvar melhor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_to_idx': self.class_to_idx
                }, save_path)
                print(f"✓ Modelo salvo com val_acc: {val_acc:.2f}%")
            
            # Ajustar learning rate
            scheduler.step(val_acc)
        
        print(f"\nTreinamento concluído! Melhor val_acc: {best_val_acc:.2f}%")
        
        # Plotar curvas
        self.plot_training_curves(history)
        
        return history
    
    def plot_training_curves(self, history: dict):
        """Plota curvas de treino"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Curvas de treino salvas em: training_curves.png")
    
    def evaluate(self, val_loader: DataLoader):
        """Avalia modelo no conjunto de validação"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=[self.idx_to_class[i] for i in range(self.num_classes)]
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[self.idx_to_class[i] for i in range(self.num_classes)],
            yticklabels=[self.idx_to_class[i] for i in range(self.num_classes)]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix salva em: confusion_matrix.png")


def main():
    """Função principal para treinar modelo"""
    
    # Configurações
    DATA_DIR = 'dataset/luminaires'  # Estrutura: dataset/luminaires/LUXA200/img1.jpg
    NUM_CLASSES = 10  # Ajustar conforme número de modelos
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Criar classificador
    classifier = LuminaireClassifier(num_classes=NUM_CLASSES, model_name='resnet50')
    
    # Preparar dados
    train_loader, val_loader = classifier.prepare_data(
        DATA_DIR, batch_size=BATCH_SIZE, val_split=0.2
    )
    
    # Treinar
    history = classifier.train(
        train_loader, val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE,
        save_path='luminaire_classifier.pth'
    )
    
    # Avaliar
    classifier.evaluate(val_loader)
    
    print("\n✓ Treinamento concluído!")


if __name__ == "__main__":
    main()