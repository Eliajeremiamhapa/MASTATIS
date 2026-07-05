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
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. MODEL YAKO YA MASTITIS
MODEL_PATH = os.path.join(BASE_DIR, 'mystatis_model.onnx')
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
class_names = ['MYSTATIS', 'NOT MYSTATIS']

# 2. MODEL YA KILINDA MLANGO (MobileNetV2)
# Pakua faili hili na uliweke folda moja na app.py
GUARD_MODEL_PATH = os.path.join(BASE_DIR, 'mobilenetv2.onnx')

if os.path.exists(GUARD_MODEL_PATH):
    guard_session = ort.InferenceSession(GUARD_MODEL_PATH, providers=['CPUExecutionProvider'])
    guard_input_name = guard_session.get_inputs()[0].name
    print("[INFO] Kilinda mlango (MobileNetV2) kimeanzishwa kikamilifu!")
else:
    guard_session = None
    print("[WARNING] Faili la 'mobilenetv2.onnx' halipatikani. Ulinzi wa picha za nje umezimwa!")

# ID za madaraja ya ImageNet yanayohusiana na Wanyama/Ng'ombe (Ng'ombe, Ndama, n.k.)
# 345: ox (fahali), 346: water buffalo, 347: bison, 397: bighorn sheep
COW_CLASSES = [345, 346, 347, 397] 

# --- LOGIC YA KEEP-ALIVE (Imeachwa kama ilivyokuwa) ---
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

threading.Thread(target=keep_alive, daemon=True).start()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded under key "file"'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Fungua picha mara moja tu
        original_img = Image.open(file).convert('RGB')

        # -------------------------------------------------------------
        # HATUA YA A: KILINDA MLANGO (ANGALIA KAMA NI NYUMBA AU KITU NYINGINE)
        # -------------------------------------------------------------
        if guard_session is not None:
            # Preprocessing maalum ya MobileNet (Kawaida inataka ImageNet normalization)
            guard_img = original_img.resize((224, 224))
            guard_array = np.array(guard_img).astype(np.float32) / 255.0
            
            # ImageNet Normalization (Mean & Std)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            guard_array = (guard_array - mean) / std
            
            # MobileNet mara nyingi inataka Format ya NCHW: [Batch, Channels, Height, Width]
            guard_array = np.transpose(guard_array, (2, 0, 1))
            guard_array = np.expand_dims(guard_array, axis=0)

            # Run Kilinda Mlango
            guard_outputs = guard_session.run(None, {guard_input_name: guard_array})
            guard_predictions = np.squeeze(guard_outputs[0])
            
            # Pata daraja lenye uwezekano mkubwa (Argmax)
            predicted_class_id = int(np.argmax(guard_predictions))
            
            # KAMA DARADA HALIHUSU WANYAMA/NG'OMBE -> KATAA HAPO HAPO
            # Unaweza pia kuongeza check ya kama ni kitu kingine kabisa
            if predicted_class_id not in COW_CLASSES:
                # Unaweza pia kuongeza unafuu: Kama mnyama hajapatikana kabisa, kataa picha ya nyumba au gari
                # Hii inasaidia kuzuia "Out of Distribution" images
                return jsonify({
                    'success': False,
                    'error': 'Mfumo umetambua picha hii haihusiani na Ng\'ombe au Mnyama. Tafadhali weka picha sahihi.'
                }), 400

        # -------------------------------------------------------------
        # HATUA YA B: RUN MODEL YAKO YA MASTITIS (Kama picha imepita ulinzi)
        # -------------------------------------------------------------
        img = original_img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # [1, 224, 224, 3]

        outputs = session.run(None, {input_name: img_array})
        predictions = np.squeeze(outputs[0]) 

        sigmoid_score = float(predictions)
        if sigmoid_score > 0.5:
            result = class_names[1]
            confidence = sigmoid_score * 100
        else:
            result = class_names[0]
            confidence = (1.0 - sigmoid_score) * 100

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
