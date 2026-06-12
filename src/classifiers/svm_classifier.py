"""
SVM Comparison Classifier
===========================
Optional comparison classifier — comparative evaluation against CNN.
Pipeline:
  Feature Vector (from any extractor)
       ↓
    SVM (RBF / Linear / Poly kernel)
       ↓
  Prediction + Confidence Score
"""

import numpy as np, os, pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


class SVMClassifier:
    def __init__(self, kernel='rbf', C=10.0, gamma='scale', probability=True):
        self.kernel = kernel; self.C = C; self.gamma = gamma
        self.probability = probability; self.pipeline = None
        self.label_encoder = LabelEncoder(); self.is_trained = False

    def build(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,
                        probability=self.probability, class_weight='balanced', random_state=42))
        ]); return self

    def train(self, X_train, y_train):
        if self.pipeline is None: self.build()
        self.pipeline.fit(X_train, self.label_encoder.fit_transform(y_train))
        self.is_trained = True; return self

    def predict(self, X):
        if not self.is_trained: raise RuntimeError("SVM not trained.")
        return self.label_encoder.inverse_transform(self.pipeline.predict(X))

    def predict_proba(self, X):
        if not self.is_trained: raise RuntimeError("SVM not trained.")
        return self.pipeline.predict_proba(X)

    def predict_single(self, feature_vector):
        X = feature_vector.reshape(1, -1)
        pred  = self.predict(X)[0]
        proba = self.predict_proba(X)[0]
        top   = np.argsort(proba)[-5:][::-1]
        conf  = float(proba[top[0]])
        level = "High" if conf >= 0.80 else ("Medium" if conf >= 0.50 else "Low")
        classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
        return {
            'class_id': int(pred), 'confidence': f"{conf*100:.2f}%",
            'confidence_level': level, 'model_used': f'SVM ({self.kernel.upper()} kernel)',
            'top_predictions': [{'class_id': int(classes[i]),
                'label': f"Person {int(classes[i])+1}",
                'confidence': f"{float(proba[i])*100:.2f}%"} for i in top]
        }

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return {'accuracy': acc, 'accuracy_pct': f"{acc*100:.2f}%",
                'report': classification_report(y_test, y_pred, zero_division=0)}

    def save(self, path='weights/svm_classifier.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'wb') as f:
            pickle.dump({'pipeline':self.pipeline,'label_encoder':self.label_encoder,
                         'kernel':self.kernel,'is_trained':self.is_trained}, f)

    def load(self, path='weights/svm_classifier.pkl'):
        if not os.path.exists(path): raise FileNotFoundError(f"Not found: {path}")
        with open(path,'rb') as f: data = pickle.load(f)
        self.pipeline=data['pipeline']; self.label_encoder=data['label_encoder']
        self.kernel=data['kernel']; self.is_trained=data['is_trained']; return self
