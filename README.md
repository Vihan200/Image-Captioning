# 🧠 Multimodal Vision-Language Project: Image Captioning & Generation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-green)
![Transformer](https://img.shields.io/badge/Transformer-Encoder%2FDecoder-yellow)
![StableDiffusion](https://img.shields.io/badge/StableDiffusion-v1.5-lightgrey)

This project explores two powerful multimodal tasks in deep learning: **generating captions for images** and **creating images from text descriptions**. Both tasks use the **Flickr8k dataset** and demonstrate how to bridge visual and textual data using neural networks.

---

## 📌 Phase 1: Image Captioning with CNN-Transformer  

🔗 [View Phase 1 Notebook on GitHub](https://github.com/kalana-mihiranga/Image-Captioning/blob/main/Phase1.ipynb)


An end-to-end deep learning pipeline that generates descriptive captions for images using a hybrid **EfficientNetB0 + Transformer Decoder** model, built entirely with TensorFlow/Keras.

### 🔑 Features
- **CNN-Transformer Architecture**: Combines CNN for image encoding and Transformer layers for sequence generation
- **Multi-Caption Learning**: Trains on five captions per image from Flickr8k
- **Masked Loss & Accuracy**: Custom loss function with padding token support
- **Learning Rate Warm-up**: Helps stabilize early-stage training

### 📊 Performance Snapshot
| Epoch | Training Acc | Training Loss | Validation Acc | Validation Loss |
|-------|--------------|----------------|----------------|-----------------|
| 96    | 0.5133       | 10.1587        | 0.4064         | 15.5985         |

---

## 🎨 Phase 2: Text-to-Image Generation with Stable Diffusion

This phase flips the direction: from **caption-to-image**. It uses **Stable Diffusion v1.5** (via Hugging Face's `diffusers`) to generate high-quality images directly from text prompts.

### 🧠 Highlights
- **Pretrained Inference**: Uses Stable Diffusion out of the box
- **Prompt-Based Generation**: Real captions serve as creative prompts
- **Zero Training Required**: Focuses on experimentation and inference
- **Diverse Image Outputs**: Covers people, animals, nature, and action scenes

### 📝 Sample Prompts
"A futuristic cityscape at night with neon lights",
"A realistic photo of a dragon flying over mountains",
"An astronaut riding a horse in space, digital art",
"A cute corgi puppy wearing sunglasses on a beach"
<p align="center">
    <img src="download (2).png" width="500px" height="750px">
</p>
### 🖼️ Output Handling
Generated images are saved using:
```python
img.save(f"generated_image_{i+1}.png")
