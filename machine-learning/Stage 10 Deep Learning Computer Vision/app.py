import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title='Plant Disease Detector', page_icon='🌿', layout='wide')

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot',
    'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Tomato___Early_blight', 'Tomato___healthy',
]

TREATMENT = {
    'Apple___Apple_scab'           : 'Apply fungicide (captan/mancozeb). Remove infected leaves.',
    'Apple___Black_rot'            : 'Prune infected branches. Apply copper-based fungicide.',
    'Apple___Cedar_apple_rust'     : 'Remove nearby cedar trees. Apply myclobutanil fungicide.',
    'Apple___healthy'              : 'No disease detected. Continue regular care.',
    'Corn___Cercospora_leaf_spot'  : 'Apply strobilurin fungicide. Improve air circulation.',
    'Corn___Common_rust'           : 'Apply fungicide early. Use rust-resistant varieties.',
    'Corn___Northern_Leaf_Blight'  : 'Apply propiconazole fungicide. Rotate crops.',
    'Corn___healthy'               : 'No disease detected. Continue regular care.',
    'Tomato___Early_blight'        : 'Remove affected leaves. Apply chlorothalonil fungicide.',
    'Tomato___healthy'             : 'No disease detected. Continue regular care.',
}

IMG_SIZE = 96

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_mobilenet.keras')

def preprocess(img_pil, size=IMG_SIZE):
    img = np.array(img_pil.convert('RGB').resize((size, size)))
    return img.astype(np.float32) / 255.0

st.title('🌿 Plant Disease Detector')
st.markdown('**AI-powered leaf disease classification for farmers**')

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('📷 Upload a Leaf Image')
    uploaded = st.file_uploader('Choose a leaf photo', type=['jpg','jpeg','png'])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption='Uploaded Leaf', use_column_width=True)

with col2:
    if uploaded:
        st.subheader('🔍 Diagnosis')
        with st.spinner('Analysing ...'):
            try:
                model = load_model()
                img   = preprocess(img_pil)
                probs = model.predict(img[np.newaxis], verbose=0)[0]
                top3  = np.argsort(probs)[::-1][:3]
                pred_class = CLASS_NAMES[top3[0]]
                pred_conf  = probs[top3[0]]
                is_healthy = 'healthy' in pred_class.lower()
                badge = '✅ Healthy' if is_healthy else '⚠️ Disease Detected'
                st.markdown(f'### {badge}')
                st.markdown(f"**Predicted:** `{pred_class.replace('___',': ').replace('_',' ')}`")
                st.metric('Confidence', f'{pred_conf:.1%}')
                st.markdown('---')
                st.markdown('### 💊 Treatment')
                st.info(TREATMENT.get(pred_class, 'Consult an agronomist.'))
                st.markdown('### 📊 Top-3 Predictions')
                for i, idx in enumerate(top3):
                    name = CLASS_NAMES[idx].replace('___',': ').replace('_',' ')
                    st.progress(float(probs[idx]), text=f'{i+1}. {name} — {probs[idx]:.1%}')
            except Exception as e:
                st.error(f'Model not found. Train model first.\n{e}')
    else:
        st.info('👆 Upload a leaf image to get started')

st.markdown('---')
st.caption('Stage 10 Portfolio — MobileNetV2 | TensorFlow + Streamlit')