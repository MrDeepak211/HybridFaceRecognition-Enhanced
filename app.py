"""
Flask Application — Hybrid Face Recognition System
====================================================
Author : Panchalwar Mam's Research — Enhanced Implementation
Features: SIFT|HOG|Gabor|Canny | Custom CNN | MobileNetV2 | SVM Comparison
          Eye-Landmark Alignment | Softmax/Sigmoid Switching | Confidence Scoring
"""

import os, cv2, json, base64, numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from src.preprocessing.processor import detect_and_align_face
from src.features.extractors import (extract_hybrid_features, extract_research_fusion,
                                      extract_by_method)
from src.classifiers.svm_classifier import SVMClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

HISTORY_FILE = 'data/recognition_history.json'

# ── Load models ──────────────────────────────────────────────
model = model_mv2 = None
svm_classifiers = {}

for path, tag in [('weights/best_hybrid_model.h5','cnn'),
                  ('weights/best_mobilenetv2_model.h5','mv2')]:
    if os.path.exists(path):
        try:
            m = tf.keras.models.load_model(path)
            if tag == 'cnn': model = m
            else: model_mv2 = m
            print(f"✅ Model loaded: {path}")
        except Exception as e:
            print(f"⚠️  {path}: {e}")

for kernel in ['rbf', 'linear']:
    p = f'weights/svm_{kernel}.pkl'
    if os.path.exists(p):
        try:
            svm_classifiers[kernel] = SVMClassifier(kernel=kernel).load(p)
            print(f"✅ SVM ({kernel}) loaded")
        except Exception as e:
            print(f"⚠️  SVM ({kernel}): {e}")

for d in ['data', app.config['UPLOAD_FOLDER'], 'weights', 'results']:
    os.makedirs(d, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f: json.dump([], f)

# ── Helpers ──────────────────────────────────────────────────
def confidence_level(conf_float):
    if conf_float >= 0.80: return "High"
    if conf_float >= 0.50: return "Medium"
    return "Low"

def save_history(record):
    try:
        with open(HISTORY_FILE) as f: h = json.load(f)
        h.insert(0, record)
        with open(HISTORY_FILE, 'w') as f: json.dump(h[:500], f, indent=2)
    except: pass

def preprocess_for_model(face, use_mv2=False):
    if use_mv2:
        rgb = cv2.cvtColor((face*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return cv2.resize(rgb,(96,96)).reshape(1,96,96,3) / 255.0
    return face.reshape(1,128,128,1)

def run_prediction(img, method='hybrid', classifier='custom',
                   activation='softmax', fusion='research'):
    """
    method     : sift | hog | gabor | canny | hybrid | research_fusion
    classifier : custom | mobilenetv2 | svm_rbf | svm_linear
    activation : softmax | sigmoid
    fusion     : research (SIFT+HOG+Gabor) | full (SIFT+HOG+Gabor+Canny)
    """
    face = detect_and_align_face(img)

    # Feature extraction
    if method == 'hybrid':
        features = (extract_research_fusion(face)
                    if fusion == 'research'
                    else extract_hybrid_features(face))
    else:
        features = extract_by_method(face, method)

    # ── SVM branch ───────────────────────────────────────
    if classifier.startswith('svm'):
        kernel = classifier.split('_')[1] if '_' in classifier else 'rbf'
        if kernel not in svm_classifiers:
            raise RuntimeError(f"SVM ({kernel}) not trained yet. Train the model first.")
        return {**svm_classifiers[kernel].predict_single(features),
                'feature_count': int(len(features)),
                'method': method, 'fusion_type': fusion,
                'activation': 'N/A (SVM)'}

    # ── CNN / MobileNetV2 branch ──────────────────────────
    use_mv2 = (classifier == 'mobilenetv2' and model_mv2 is not None)
    active  = model_mv2 if use_mv2 else model
    if active is None:
        raise RuntimeError("No trained model found. Please train first.")

    img_input  = preprocess_for_model(face, use_mv2)
    feat_input = features.reshape(1, -1)

    raw = active.predict(
        {'image_input': img_input, 'feature_input': feat_input}, verbose=0
    )
    logits = raw[0] if isinstance(raw, list) else raw
    logits = logits[0]

    # ── Softmax / Sigmoid switching ───────────────────────
    if activation == 'sigmoid':
        probs = 1.0 / (1.0 + np.exp(-logits))   # element-wise sigmoid
        probs = probs / probs.sum()               # normalise to sum=1
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()                       # stable softmax

    top_idx = np.argsort(probs)[-5:][::-1]
    conf_f  = float(probs[top_idx[0]])

    return {
        'class_id':         int(top_idx[0]),
        'confidence':       f"{conf_f*100:.2f}%",
        'confidence_level': confidence_level(conf_f),   # ← explicit High/Medium/Low
        'top_predictions':  [{'class_id': int(i), 'label': f"Person {int(i)+1}",
                              'confidence': f"{float(probs[i])*100:.2f}%"} for i in top_idx],
        'feature_count':    int(len(features)),
        'method':           method,
        'fusion_type':      fusion,
        'activation':       activation,
        'model_used':       'MobileNetV2' if use_mv2 else 'Custom CNN',
        'feature_breakdown': {'SIFT':'128-dim','HOG':'~8100-dim',
                              'Gabor':'24-dim','Canny':'65-dim'},
    }

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        model_loaded=(model is not None),
        mv2_loaded=(model_mv2 is not None),
        svm_loaded=bool(svm_classifiers))

@app.route('/predict', methods=['POST'])
def predict():
    if not request.files.get('file'):
        return jsonify({'error': 'No file uploaded'}), 400
    file       = request.files['file']
    method     = request.form.get('method', 'hybrid')
    classifier = request.form.get('classifier', 'custom')
    activation = request.form.get('activation', 'softmax')
    fusion     = request.form.get('fusion', 'research')
    filename   = secure_filename(file.filename)
    filepath   = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        img    = cv2.imread(filepath)
        result = run_prediction(img, method, classifier, activation, fusion)
        save_history({**result, 'timestamp': datetime.now().isoformat(), 'source': 'upload'})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera-capture', methods=['POST'])
def camera_capture():
    try:
        data       = request.get_json()
        img_data   = data.get('image','').split(',')[-1]
        method     = data.get('method','hybrid')
        classifier = data.get('classifier','custom')
        activation = data.get('activation','softmax')
        fusion     = data.get('fusion','research')
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error':'Invalid image'}), 400
        result = run_prediction(img, method, classifier, activation, fusion)
        save_history({**result, 'timestamp': datetime.now().isoformat(), 'source': 'camera'})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    files      = request.files.getlist('files[]')
    method     = request.form.get('method','hybrid')
    classifier = request.form.get('classifier','custom')
    activation = request.form.get('activation','softmax')
    fusion     = request.form.get('fusion','research')
    results, errors = [], []
    for file in files:
        try:
            fn = secure_filename(file.filename)
            fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            file.save(fp)
            res = run_prediction(cv2.imread(fp), method, classifier, activation, fusion)
            results.append({**res, 'filename': fn})
        except Exception as e:
            errors.append({'filename': file.filename, 'error': str(e)})
    return jsonify({'results':results,'errors':errors,
                    'success':len(results),'failed':len(errors)})

@app.route('/api/history')
def get_history():
    page = request.args.get('page',1,int)
    limit= request.args.get('limit',20,int)
    try:
        with open(HISTORY_FILE) as f: h = json.load(f)
        s = (page-1)*limit
        return jsonify({'records':h[s:s+limit],'total':len(h),'page':page})
    except:
        return jsonify({'records':[],'total':0,'page':page})

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    with open(HISTORY_FILE,'w') as f: json.dump([],f)
    return jsonify({'message':'History cleared'})

@app.route('/api/stats')
def get_stats():
    try:
        with open(HISTORY_FILE) as f: h = json.load(f)
        confs = [float(r['confidence'].replace('%','')) for r in h if 'confidence' in r]
        avg   = sum(confs)/len(confs) if confs else 0
        return jsonify({
            'totalRecognitions': len(h),
            'averageConfidence': f"{avg:.2f}%",
            'personsCount': len(set(r.get('class_id',0) for r in h)),
            'modelLoaded': model is not None,
            'mv2Loaded':   model_mv2 is not None,
            'svmLoaded':   bool(svm_classifiers),
        })
    except Exception as e:
        return jsonify({'totalRecognitions':0,'averageConfidence':'0%',
                        'personsCount':0,'modelLoaded':False,'mv2Loaded':False,'svmLoaded':False})

@app.route('/api/comparison-results')
def comparison_results():
    return jsonify({
        'paper_results': [
            {'method':'SIFT + CNN','optimizer':'Adam','activation':'Softmax',
             'orl_acc':'92.50%','sheffield_acc':'100.00%','best':True},
            {'method':'HOG + CNN','optimizer':'Adamax','activation':'Softmax',
             'orl_acc':'91.25%','sheffield_acc':'97.50%','best':False},
            {'method':'Gabor + CNN','optimizer':'Adam','activation':'Softmax',
             'orl_acc':'93.75%','sheffield_acc':'95.00%','best':False},
            {'method':'Canny + CNN','optimizer':'SGD','activation':'Sigmoid',
             'orl_acc':'80.00%','sheffield_acc':'82.50%','best':False},
        ],
        'proposed': [
            {'method':'Research Fusion (SIFT+HOG+Gabor+CNN)','status':'Under Evaluation'},
            {'method':'Full Hybrid (SIFT+HOG+Gabor+Canny+CNN)','status':'Under Evaluation'},
            {'method':'MobileNetV2 Transfer Learning','status':'Under Evaluation'},
            {'method':'SVM (RBF kernel)','status':'Under Evaluation'},
            {'method':'SVM (Linear kernel)','status':'Under Evaluation'},
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
