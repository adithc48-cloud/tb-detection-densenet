import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Page Setup
st.set_page_config(
    page_title="Tuberculosis Diagnostic System",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 Tuberculosis Detection & Visual Explainability")
st.markdown(
    "Automated triage using **DenseNet-121**, **CLAHE contrast enhancement**, and **Grad-CAM**."
)

# Sidebar Objectives
with st.sidebar:
  st.header("📌 Project Objectives")
  st.markdown("""
    * **Objective 1:** Open-access public CXR benchmark dataset.
    * **Objective 2:** Transfer learning via pre-trained DenseNet-121.
    * **Objective 3:** Standard metrics: Sensitivity, Specificity & ROC-AUC.
    * **Objective 4:** Interactive web interface with real-time explainability.
    """)
  st.divider()
  cam_alpha = st.slider("Heatmap Blend Opacity", 0.1, 0.9, 0.5, 0.05)

# CPU Runtime Model Cache
device = torch.device("cpu")


@st.cache_resource
def load_model():
  model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
  model.classifier = torch.nn.Sequential(
      torch.nn.Dropout(0.3), torch.nn.Linear(model.classifier.in_features, 2)
  )
  model.to(device)
  model.eval()
  return model


model = load_model()
target_layers = [model.features.denseblock4.denselayer16.conv2]
cam = GradCAM(model=model, target_layers=target_layers)

# Image Ingestion
uploaded_file = st.file_uploader(
    "Upload Chest Radiograph (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
  resized_rgb = cv2.resize(raw_rgb, (256, 256))

  # CLAHE Preprocessing
  gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  enhanced_gray = clahe.apply(gray)
  enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

  # Tensor Normalization
  norm_float = enhanced_rgb.astype(np.float32) / 255.0
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  normalized_tensor = (norm_float - mean) / std
  input_tensor = (
      torch.tensor(normalized_tensor)
      .permute(2, 0, 1)
      .unsqueeze(0)
      .float()
      .to(device)
  )

  # Inference
  with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    tb_prob = float(probs[1])
    normal_prob = float(probs[0])

  # Grad-CAM Generation
  targets = [ClassifierOutputTarget(1)]
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
  heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
  heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  dynamic_overlay = np.clip(
      (cam_alpha * heatmap_rgb) + ((1.0 - cam_alpha) * norm_float), 0.0, 1.0
  )

  # 3-Stage Inspection Triad
  st.divider()
  col1, col2, col3 = st.columns(3)
  col1.image(
      enhanced_rgb, caption="1. CLAHE Enhanced", use_container_width=True
  )
  col2.image(
      grayscale_cam,
      caption="2. Activation Map",
      clamp=True,
      use_container_width=True,
  )
  col3.image(
      dynamic_overlay,
      caption=f"3. Grad-CAM Overlay ({cam_alpha:.2f})",
      use_container_width=True,
  )

  # Evaluation Metrics Output
  st.divider()
  res1, res2 = st.columns([2, 1])
  with res1:
    if tb_prob >= 0.5:
      st.error(
          f"### ⚠️ Assessment: POSITIVE for Tuberculosis\nConfidence:"
          f" **{tb_prob * 100:.2f}%**"
      )
    else:
      st.success(
          f"### ✅ Assessment: NORMAL / Healthy\nConfidence:"
          f" **{normal_prob * 100:.2f}%**"
      )
  with res2:
    st.metric("TB Probability", f"{tb_prob * 100:.1f}%")
    st.metric("Normal Baseline", f"{normal_prob * 100:.1f}%")
  st.progress(tb_prob)
