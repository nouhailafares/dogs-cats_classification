# 🐾 Dogs vs Cats Classifier — Deep Learning Project

## Description
Ce projet est une application de classification d’images utilisant le Deep Learning pour distinguer les **chiens** et les **chats**.

Deux approches ont été comparées :
- CNN from scratch
- Data augmentation
- Transfer Learning avec **MobileNetV2**

L’objectif est d’améliorer les performances de classification sur un dataset d’images Dogs vs Cats.

---

## Dataset
Dataset utilisé :
- Kaggle Dogs vs Cats
- Images de chiens et chats classées en dossiers `train/` et `validation/`

---

## Modèles utilisés

### 1. CNN from scratch
- Convolution2D
- MaxPooling2D
- Dropout
- Dense layers

### 2. Data Augmentation
- Rotation
- Zoom
- Translation
- Flip horizontal

### 3. Transfer Learning (MobileNetV2)
- Modèle pré-entraîné sur ImageNet
- Fine-tuning de la tête de classification
- GlobalAveragePooling2D + Dense layers

---

## Résultats

- CNN from scratch : ~72% accuracy
- Data augmentation : amélioration limitée
- MobileNetV2 : ~98% accuracy

Le transfer learning donne les meilleures performances grâce aux features pré-apprises.

---

## Application Web

Une application **Streamlit** a été développée pour permettre :
- Upload d’image
- Prédiction en temps réel
- Affichage de la classe (chien ou chat)
- Score de confiance
