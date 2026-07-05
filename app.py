import os
import numpy as np
import onnxruntime as ort
import threading
import time
import requests
from Flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
 
app = Flask(__name__)
CORS(app)  # Muhimu sana kwa ajili ya Mobile App na Web integration yako
 
# 1. MFUMO WA DYNAMIC PATH KWA AJILI YA RENDER (HAKUNA D:\ TENA)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mystatis_model.onnx')
 
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Faili la model halipatikani kwenye njia hii: {MODEL_PATH}")
    print("[HINT] Hakikisha faili la 'mystatis_model.onnx' lipo kwenye folda moja na hii app.py")
 
# Anzisha ONNX Inference Session (CPU Execution pekee kwa ajili ya Render Free Tier)
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
 
# 2. LABELS ZA MODEL YA MYSTATIS (Binary: 0 ni MYSTATIS, 1 ni NOT MYSTATIS)
class_names = ['MYSTATIS', 'NOT MYSTATIS']
 
# --- LOGIC YA KUAMSHA SERVER (KEEP-ALIVE) ---
def keep_alive():
    """Inapiga picha server kila baada ya dakika 10 kuzuia isilale kwenye Render (Free Tier)"""
    # Subiri sekunde 10 mwanzoni ili kuhakikisha server imekamilisha kuwaka kikamilifu
    time.sleep(10)
    while True:
        try:
            host = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
            if host:
                url = f"https://{host}/health"
                requests.get(url, timeout=5)
                print("Keep-alive: Ping sent successfully!")
            else:
                print("Keep-alive: Waiting for Render environment...")
        except Exception as e:
            print(f"Keep-alive error: {e}")
        time.sleep(600)  # Dakika 10
 
# SULUHISHO LA UHAKIKA: Anzisha thread hapa juu moja kwa moja ili iwake chini ya Gunicorn au Python run
keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()
 
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "I am awake!"}), 200
# --------------------------------------------
 
# MWONGOZO WA NJIA KUU (HOME API ENDPOINT)
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to Elixa MYSTATIS Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image file with key 'file'",
            "/health": "GET - Check API status"
        }
    }), 200
 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded under key "file"'}), 400
         
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
 
        # 3. PREPROCESSING (KERAS SYSTEM SYSTEM)
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
         
        # Geuza kwenda numpy array na fanya Rescaling ya (1./255) kama ilivyokuwa kwenye Keras training
        img_array = np.array(img).astype(np.float32) / 255.0
         
        # TENSORFLOW FORMAT: [Batch, Height, Width, Channels] -> [1, 224, 224, 3]
        img_array = np.expand_dims(img_array, axis=0)
 
        # 4. RUN INFERENCE KWA ONNX
        outputs = session.run(None, {input_name: img_array})
        predictions = np.squeeze(outputs[0])  # Inatoa single value ya sigmoid raw output
 
        # 5. KOKOTOA MAJIBU (SIGMOID BINARY CONFIGURATION)
        sigmoid_score = float(predictions)
         
        if sigmoid_score > 0.5:
            # Karibu zaidi na 1 -> NOT MYSTATIS
            result = class_names[1]
            confidence = sigmoid_score * 100
        else:
            # Karibu zaidi na 0 -> MYSTATIS
            result = class_names[0]
            confidence = (1.0 - sigmoid_score) * 100
 
        # -------------------------------------------------------------
        # TUNALINDA MFUMO: KAMA PICHA SIO YA CHUCHU (CONFIDENCE CHINI YA 85%)
        # -------------------------------------------------------------
        if confidence < 85.0:
            return jsonify({
                'success': False,
                'error': "Tafadhali tuma picha sahihi ya chuchu za ng'ombe."
            }), 400
        # -------------------------------------------------------------
 
        # RETURNING STANDARDIZED API JSON RESPONSE FOR MOBILE APP
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        }), 200
         
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
