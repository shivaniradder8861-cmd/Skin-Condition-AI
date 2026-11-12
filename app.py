
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

st.set_page_config(page_title="Skin Condition AI", page_icon="🔬")

st.title("AI-Based Skin Condition Classification")
st.write("Skin image classification using MobileNetV2 Transfer Learning")

device = torch.device("cpu")

# Load class names
with open("model/classes.json", "r") as f:
    classes = json.load(f)

# Load trained model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )
    model.load_state_dict(
        torch.load("model/skin_vit.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict Skin Condition"):
        with st.spinner("Analyzing image..."):
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            prediction = classes[predicted.item()]

        st.success(f"Predicted category: {prediction}")
        st.write(f"Model confidence: {confidence.item()*100:.2f}%")

st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)
