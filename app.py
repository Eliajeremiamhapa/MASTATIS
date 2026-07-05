import os
import numpy as np
import onnxruntime as ort
import threading
import time
import requests
from flask import Flask, request, jsonify
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
 

# =========================================================================
# 2. MFUMO WA KILINDA MLANGO (AUTOMATIC MOBILENETV2 DOWNLOAD & LOAD)
# =========================================================================
GUARD_MODEL_PATH = os.path.join(BASE_DIR, 'mobilenetv2.onnx')
MOBILENET_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenetview/model/mobilenetv2-7.onnx"

def download_guard_model():
    """Inapakua model ya MobileNetV2 kiotomatiki kama haipo kwenye seva ya Render"""
    if not os.path.exists(GUARD_MODEL_PATH):
        print("[INFO] Tunapakua kilinda mlango (MobileNetV2 ONNX)... Tafadhali subiri.")
        try:
            response = requests.get(MOBILENET_URL, stream=True, timeout=30)
            if response.status_code == 200:
                with open(GUARD_MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("[SUCCESS] Kilinda mlango kimepakuliwa kikamilifu!")
            else:
                print(f"[ERROR] Imeshindwa kupakua model. Status code: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Hitilafu imetokea wakati wa kupakua kilinda mlango: {e}")

download_guard_model()

if os.path.exists(GUARD_MODEL_PATH):
    guard_session = ort.InferenceSession(GUARD_MODEL_PATH, providers=['CPUExecutionProvider'])
    guard_input_name = guard_session.get_inputs()[0].name
    print("[INFO] Kilinda mlango kiko tayari kulinda mfumo!")
else:
    guard_session = None
    print("[WARNING] Kilinda mlango hakikuweza kuwashwa. Picha zote zitapita moja kwa moja.")

# ID Maalum za ImageNet za Ng'ombe pekee (Ox/Fahali, Water Buffalo, Bison)
ALLOWED_COW_CLASSES = [345, 346, 347]
# =========================================================================


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
        "version": "1.1.0",
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
        original_img = Image.open(file).convert('RGB')
        
        # -------------------------------------------------------------
        # HATUA YA ULINZI: KILINDA MLANGO (KUKATAA PICHA ZISIZO ZA NG'OMBE)
        # -------------------------------------------------------------
        if guard_session is not None:
            guard_img = original_img.resize((224, 224))
            guard_array = np.array(guard_img).astype(np.float32) / 255.0
            
            # MobileNet Inahitaji normalization ya ImageNet standard standard
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            guard_array = (guard_array - mean) / std
            
            # Badili kwenda NCHW format inayotakiwa na MobileNet ONNX
            guard_array = np.transpose(guard_array, (2, 0, 1))
            guard_array = np.expand_dims(guard_array, axis=0)

            guard_outputs = guard_session.run(None, {guard_input_name: guard_array})
            guard_predictions = np.squeeze(guard_outputs[0])
            
            predicted_class_id = int(np.argmax(guard_predictions))
            
            # KAMA PICHA HAIHUSIANI NA NG'OMBE -> KATAA PAPO HAPO KULINDA MFUMO
            if predicted_class_id not in ALLOWED_COW_CLASSES:
                return jsonify({
                    'success': False,
                    'error': 'Tafadhali tuma picha sahihi ya chuchu za ng\'ombe.'
                }), 400
        # -------------------------------------------------------------

        # ENDELEA NA PROCESSING YA MODEL YAKO YA MASTITIS KAMA KAWAIDA
        img = original_img.resize((224, 224))
         
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
