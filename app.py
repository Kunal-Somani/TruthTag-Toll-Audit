import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Cross-Modal Verifier", layout="wide")

st.sidebar.title("Cross-Modal Vehicle Verification System")
st.sidebar.subheader("Team: B-Team")
st.sidebar.markdown("""
**About the Project:**
This system detects FASTag toll fraud by comparing the declared RFID tag class with the actual visual class of the vehicle. 

If a truck enters a toll using a car FASTag to pay less, this cross-modal verification system flags it automatically!

*Classical CV only — no neural networks.*
""")

demo_mode = st.sidebar.toggle("Demo Mode (No model files needed)", value=False)

if demo_mode:
    st.warning("DEMO MODE — simulated results only")

# ---------------------------------------------------------
# Caching Model & Scaler Loading
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """Load the HOG-SVM model and the scaler."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), "models", "hog_svm_model.joblib")
        scaler_path = os.path.join(os.path.dirname(__file__), "models", "scaler.joblib")
        
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return clf, scaler
    except Exception as e:
        return None, None

clf, scaler = None, None
if not demo_mode:
    clf, scaler = load_models()
    if clf is None or scaler is None:
        st.error("Failed to load models. Make sure hog_svm_model.joblib and scaler.joblib are in the models/ folder. Or enable Demo Mode in the sidebar.")

# ---------------------------------------------------------
# HOG Extraction Logic
# ---------------------------------------------------------
# HOG params (must match training)
WIN_SIZE = (64, 128)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9
CLASS_ORDER = ["Car", "Bus", "Truck"]

def get_hog_descriptor():
    return cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)

def extract_features(img_array):
    """Preprocess image and extract HOG features"""
    # Convert to grayscale if it's color
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
        
    # Resize to exactly 64x128
    resized = cv2.resize(gray, WIN_SIZE, interpolation=cv2.INTER_AREA)
    
    # Compute HOG
    hog = get_hog_descriptor()
    hist = hog.compute(resized)
    return hist.reshape(1, -1)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

# LEFT PANEL
with col1:
    st.header("Input Panel")
    
    uploaded_file = st.file_uploader("Upload Vehicle Image from Toll Camera", type=["jpg", "png", "jpeg"])
    
    rfid_claim = st.selectbox("RFID Declared Class", ["Car", "Bus", "Truck"])
    
    run_check = st.button("Run Fraud Check", type="primary", use_container_width=True)
    
    with st.expander("How this works"):
        st.markdown("""
        * **Image Preprocessing:** The image is converted to grayscale and resized to a standard 64x128.
        * **HOG Feature Extraction:** We extract Histogram of Oriented Gradients (HOG) features to capture the vehicle's shape and edges.
        * **Linear SVM Classification:** A Support Vector Machine classifies the HOG features into Car, Bus, or Truck.
        * **Cross-Modal Verification:** If the visual prediction doesn't match the RFID FASTag class, fraud is flagged!
        """)

# RIGHT PANEL
with col2:
    if run_check:
        st.header("Verification Results")
        
        if uploaded_file is None and not demo_mode:
            st.error("Please upload an image first!")
        else:
            # Display uploaded image
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                # In demo mode without upload, create a dummy placeholder image
                image = Image.new('RGB', (400, 300), color=(150, 150, 150))
            
            st.image(image, width=400, caption="Captured Toll Camera Image")
            
            st.markdown("---")
            st.subheader("Pipeline Visualization")
            
            step1, step2, step3, step4, step5, step6 = st.columns(6)
            with step1: st.info("📷\nImage")
            with step2: st.info("HOG\nFeatures")
            with step3: st.info("SVM\nClassifier")
            with step4: st.info("👁️\nVisual Class")
            with step5: st.info("🔀\nCross Check")
            with step6: st.info("⚖️\nVerdict")
            
            st.markdown("---")
            
            # Inference process
            if demo_mode:
                import time
                with st.spinner("Simulating pipeline..."):
                    time.sleep(1) # Fake processing time
                    
                # Hardcoded scenarios for demo
                if rfid_claim == "Car":
                    pred_name = "Car"
                    conf_score = 92.5
                elif rfid_claim == "Truck":
                    pred_name = "Car" # Scenario: Truck using a Car tag
                    conf_score = 88.0
                else: # Bus
                    pred_name = "Bus"
                    conf_score = 95.1
            else:
                try:
                    img_array = np.array(image.convert('RGB'))
                    features = extract_features(img_array)
                    
                    # Scale features
                    scaled_features = scaler.transform(features)
                    
                    # Predict using decision function
                    scores = clf.decision_function(scaled_features)
                    if scores.ndim == 1:
                        scores = scores.reshape(1, -1)
                        
                    probs = softmax(scores[0])
                    pred_idx = int(np.argmax(probs))
                    pred_name = CLASS_ORDER[pred_idx]
                    
                    # Confidence score 0-100%
                    conf_score = float(probs[pred_idx] * 100)
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.stop()
            
            match_status = (pred_name == rfid_claim)
            
            # Verdict Card
            if not match_status:
                st.markdown(
                    """
                    <div style="background-color: #ff4b4b; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h2 style="color: white; margin: 0;">⚠ FRAUD DETECTED</h2>
                    </div>
                    <br>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: #00cc66; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h2 style="color: white; margin: 0;">✓ VERIFIED CLEAN</h2>
                    </div>
                    <br>
                    """, unsafe_allow_html=True
                )
                
            # Metrics
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Visual Prediction", pred_name)
            met2.metric("RFID Claim", rfid_claim)
            met3.metric("Match Result", "MATCH" if match_status else "MISMATCH")
            met4.metric("Confidence Score", f"{conf_score:.1f}%")
            
            if not match_status:
                st.error(f"**Estimated toll evasion category:** {rfid_claim} tag used instead of {pred_name}")
