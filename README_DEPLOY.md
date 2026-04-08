# Deployment Guide: Cross-Modal Vehicle Verification System

This application provides an interactive web interface for the FASTag toll fraud detection system.

## Setup Instructions

### 1. Clone or Download the Repository
Clone this directory and navigate to it in your terminal:
```bash
cd tiet-ucs532p-bteam
```

### 2. Install Dependencies
It is recommended to use a virtual environment. Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Place Model Files
Ensure the trained model files are located in the `models/` folder at the root of the project:
* `models/hog_svm_model.joblib`
* `models/scaler.joblib`

*(If these files are missing, the app will fail to load the system unless 'Demo Mode' is enabled in the sidebar).*

### 4. Run the Application locally
Execute the Streamlit application:
```bash
streamlit run app.py
```
This will automatically open the web UI in your default browser at `http://localhost:8501`.

### 5. Presenting to Your Professor (Local Area Network)
If you want to demo the system from your laptop to your professor's device over the same Wi-Fi network, start the Streamlit server using:
```bash
streamlit run app.py --server.port 8501
```
Check your terminal output for the `Network URL` (e.g., `http://192.168.1.100:8501`) and open that link on the target device. 

## Demo Mode
If you need to show the full UI flow but do not have the trained model files available or cannot easily upload real toll camera images on the spot, toggle the **Demo Mode** checkbox in the sidebar. This will instantly simulate the HOG + SVM pipeline and show both a clean verification case and a cross-modal fraud verification case using hardcoded outputs without running actual inference.
