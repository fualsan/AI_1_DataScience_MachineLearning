from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import uvicorn
import joblib
from imblearn.over_sampling import SMOTE

# randomforest.py'den fonksiyonları içe aktar
from randomforest import load_data, initial_eda, preprocess_data, train_model, feature_importance, evaluate_model

# FastAPI uygulamasını tanımla
app = FastAPI()

# Eğitilmiş özellikleri saklamak için global değişken tanımla
trained_features = None

# İstek gövdesi modelini tanımla
class TrainRequest(BaseModel):
    n_estimators: int

# Eğitim uç noktası
@app.post("/train")
async def train(data: TrainRequest):
    global trained_features  # Bu değişkenin global olarak kullanılacağını belirtiyoruz

    # Veriyi yükle
    file_path = 'income_evaluation.csv'
    df = load_data(file_path)

    # İlk EDA'yı gerçekleştir
    initial_eda(df)

    # Veriyi ön işle
    X_train, X_test, y_train, y_test = preprocess_data(df)
    trained_features = X_train.columns.tolist()

    # Modeli eğit
    model = train_model(X_train, X_test, y_train, y_test, n_estimators=data.n_estimators)

    # Özellik önemini al
    feature_scores = feature_importance(model, X_train)
    #least_important_feature = feature_scores.idxmin()

    # Model değerlendirmesini yap
    evaluate_model(model, X_test, y_test)

    # MLflow deneyini kur
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("income_classifier")

    # MLflow çalışmasını başlat
    with mlflow.start_run():

        # Parametreleri kaydet
        mlflow.log_param("n_estimators", data.n_estimators)
        mlflow.log_param("random_state", 0)

        # Test seti üzerinde tahmin yap
        y_pred = model.predict(X_test)

        # Sınıflandırma raporu oluştur
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # Metrikleri kaydet
        mlflow.log_metric("accuracy", classification_rep['accuracy'])
        mlflow.log_metric("precision", classification_rep['weighted avg']['precision'])
        mlflow.log_metric("recall", classification_rep['weighted avg']['recall'])
        mlflow.log_metric("f1-score", classification_rep['weighted avg']['f1-score'])


        # Karışıklık matrisini kaydet
        cm_fig = plt.figure()
        cm_matrix = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                                 columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.title("Confusion Matrix")
        plt.close(cm_fig)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        # Özellik önemini görselleştir
        feat_importance_fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_scores[:10], y=feature_scores.index[:10])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.xticks(rotation=45)  # x etiketlerini 45 derece döndür
        plt.tight_layout()  # Daha iyi yerleşim için
        plt.close(feat_importance_fig)
        mlflow.log_figure(feat_importance_fig, "feature_importance.png")
        
        # Sınıflandırma raporunu kaydet
        class_report = classification_report(y_test, y_pred)
        mlflow.log_text(class_report, "classification_report.txt")

        # Modeli kaydet
        mlflow.sklearn.log_model(model, "random_forest_model")

    return {"message": "Model eğitildi ve MLflow ile kaydedildi."}

# Tahmin istek gövdesi modelini tanımla
class PredictRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Tahmin uç noktası
@app.post("/predict")
async def predict(run_id: str, data: PredictRequest):
    # Tahmin için girdi verisini hazırla
    input_data = pd.DataFrame([data.model_dump()]) #eskiden dict() idi
    global trained_features  # Eğitim sırasında kullanılan özellik adlarını global olarak al

    # Deney kimliği ve Çalışma kimliği ile eğitilmiş modeli yükle
    logged_model = f"runs:/{run_id}/random_forest_model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Kodlayıcıyı yükle
    encoder = joblib.load('encoder.joblib')

    # Yüklenen kodlayıcıyı kullanarak girdi verisini ön işle
    input_data_encoded = encoder.transform(input_data)

    # Tahmin yap
    prediction = loaded_model.predict(input_data_encoded)

    # Tahmini listeye dönüştür
    prediction_list = prediction.tolist()

    return {"prediction": prediction_list}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
