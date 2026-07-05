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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. MFUMO WA DYNAMIC PATH KWA AJILI YA MODEL YAKO YA MASTITIS
MODEL_PATH = os.path.join(BASE_DIR, 'mystatis_model.onnx')

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Faili la model halipatikani kwenye njia hii: {MODEL_PATH}")
    print("[HINT] Hakikisha faili la 'mystatis_model.onnx' lipo kwenye folda moja na hii app.py")

# Anzisha ONNX Inference Session ya Mastitis
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
class_names = ['MYSTATIS', 'NOT MYSTATIS']


# 2. MFUMO WA KILINDA MLANGO (AUTOMATIC MOBILENETV2 DOWNLOAD & LOAD)
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

# ID za ImageNet ambazo NI NG'OMBE PEEKEE (Ox/Fahali, Water Buffalo, Bison). 
# Picha ISIPOANGUKIA humu, inakataliwa papo hapo ili kulinda mfumo dhidi ya vitanda, nyumba au chumba.
ALLOWED_COW_CLASSES = [345, 346, 347]


# --- LOGIC YA KUAMSHA SERVER (KEEP-ALIVE) ---
def keep_alive():
    time.sleep(10)
    while True:
        try:
            host = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
            if host:
                url = f"https://{host}/health"
                requests.get(url, timeout=5)
        except Exception:
            pass
        time.sleep(600)

keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "I am awake!"}), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to Elixa Mastitis Detection API (With Guard Protection)",
        "version": "1.3.0",
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

        original_img = Image.open(file).convert('RGB')

        # -------------------------------------------------------------
        # HATUA YA A: KILINDA MLANGO (WHITELISTING YA NG'OMBE PEEKEE)
        # -------------------------------------------------------------
        if guard_session is not None:
            guard_img = original_img.resize((224, 224))
            guard_array = np.array(guard_img).astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            guard_array = (guard_array - mean) / std
            
            guard_array = np.transpose(guard_array, (2, 0, 1))
            guard_array = np.expand_dims(guard_array, axis=0)

            guard_outputs = guard_session.run(None, {guard_input_name: guard_array})
            guard_predictions = np.squeeze(guard_outputs[0])
            
            predicted_class_id = int(np.argmax(guard_predictions))
            
            # KAMA PICHA SIO YA NG'OMBE/CHUCHU ZAKE (HAIPO KWENYE ALLOWED LIST) -> KATAA PAPO HAPO
            if predicted_class_id not in ALLOWED_COW_CLASSES:
                return jsonify({
                    'success': False,
                    'error': 'Tafadhali tuma picha sahihi.'
                }), 400

        # -------------------------------------------------------------
        # HATUA YA B: RUN MODEL YAKO YA MASTITIS (Kama imepita ulinzi wa ng'ombe)
        # -------------------------------------------------------------
        img = original_img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        outputs = session.run(None, {input_name: img_array})
        predictions = np.squeeze(outputs[0])  

        sigmoid_score = float(predictions)
        
        if sigmoid_score > 0.5:
            result = class_names[1]
            confidence = sigmoid_score
        else:
            result = class_names[0]
            confidence = 1.0 - sigmoid_score

        # ULINZI WA NYONGEZA: Kama confidence ya model yako ipo chini ya 90% (inaashiria picha haina sifa kamili za mastitis/not mastitis), ikatae pia.
        if (confidence * 100) < 90.0:
            return jsonify({
                'success': False,
                'error': 'Tafadhali tuma picha sahihi.'
            }), 400

        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f"{(confidence * 100):.2f}%"
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
