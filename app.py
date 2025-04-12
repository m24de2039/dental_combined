import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

from classification_utils import load_model as load_clf_model, predict_category
from segmentation_utils import load_model as load_seg_model, predict_mask

# -- Page config --
st.set_page_config(page_title="Dental AI Suite", layout="wide")

# -- App Title --
st.markdown("<h1 style='text-align:center;'>🦷 Dental AI Suite</h1>", unsafe_allow_html=True)
st.markdown("---")

# -- Group Info Box --
st.markdown("""
<div style='padding:15px; border:2px solid #dee2e6; border-radius:10px; background-color:#f8f9fa'>
    <h4>👨‍💻 Project by Group 23 - PGDDE (IIT Jodhpur)</h4>
    <ul>
        <li><b>Bindesh Chauhan</b> (M24DE2004)</li>
        <li><b>Gaurav Ranjan</b> (M24DE2006)</li>
        <li><b>Kironmoy Dhali</b> (M24DE2013)</li>
        <li><b>Vishal Singh</b> (M24DE2039)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -- App selector --
task = st.radio("Choose Task", ["🧠 Dental Disease Classification", "🧬 Dental Image Segmentation"])

# -- Upload image --
uploaded_file = st.file_uploader("📤 Upload a dental image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

    with col2:
        if task == "🧠 Dental Disease Classification":
            clf_model = load_clf_model("models/iitj_dental_cnn.pth")
            with st.spinner("⏳ Predicting..."):
                prediction = predict_category(clf_model, image)
            st.markdown(f"""
            <div style='padding:20px; border-radius:10px; background-color:#e6f4ea; border: 2px solid #34a853;'>
                <h3 style='color:#0b6623;'>✅ Prediction: <em>{prediction}</em></h3>
            </div>
            """, unsafe_allow_html=True)

        elif task == "🧬 Dental Image Segmentation":
            seg_model = load_seg_model("models/best_model.pth")
            with st.spinner("⏳ Segmenting..."):
                mask = predict_mask(seg_model, image)
            st.subheader("🧠 Predicted Mask")
            fig, ax = plt.subplots()
            ax.imshow(image, alpha=0.7)
            ax.imshow(mask, cmap='jet', alpha=0.3)
            ax.axis('off')
            st.pyplot(fig)

    st.success("✅ Done!")

else:
    st.info("Please upload a dental image to get started.")

# -- Footer --
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>© 2025 Group 23 | PGDDE - IIT Jodhpur</p>", unsafe_allow_html=True)
