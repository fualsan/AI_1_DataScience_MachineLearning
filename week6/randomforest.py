# Bu kütüphaneler, gerekli işlevleri gerçekleştirmek için kullanılır.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
from imblearn.over_sampling import SMOTE

# Uyarıları kapatma
warnings.filterwarnings('ignore')

# Veri setini yükleme işlevi
def load_data(file_path):
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df = pd.read_csv(file_path)
    df.columns = col_names
    return df

# İlk Veri Keşfi ve Analizi (EDA) işlevi
def initial_eda(df):
    if isinstance(df, pd.DataFrame):
        total_na = df.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = df.columns
        dtyp = df.dtypes
        uniq = df.nunique()
        na_val = df.isna().sum()
        for i in range(len(df.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))        
    else:
        print("Expect a DataFrame but got a %15s" % (type(df)))

# Veri ön işleme işlevi
def preprocess_data(df):
    df['workclass'].replace(' ?', np.NaN, inplace=True)
    df['occupation'].replace(' ?', np.NaN, inplace=True)
    df['native_country'].replace(' ?', np.NaN, inplace=True)
    X = df.drop(['income'], axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    for df2 in [X_train, X_test]:
        df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
        df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
        df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)
    encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                     'race', 'sex', 'native_country'])
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    joblib.dump(encoder, 'encoder.joblib')

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    cols = X_train_encoded.columns
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=cols)

    # SMOTE uygulama
    smote = SMOTE(sampling_strategy={' <=50K': 24720, ' >50K': 24720 },random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled_df, y_train)

    return X_train_balanced, X_test_scaled_df, y_train_balanced, y_test

# Model eğitimi işlevi
def train_model(X_train, X_test, y_train, y_test, n_estimators=10):
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy score with {} decision-trees : {:.4f}'.format(n_estimators, accuracy))
    return rfc

# Öznitelik önemini hesaplayan işlev
def feature_importance(model, X_train):
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return feature_scores


# Modeli değerlendiren işlev
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy score: {0:0.4f}'.format(accuracy))
    print(classification_report(y_test, y_pred))

# Ana işlev
def main():
    # Veriyi yükle
    file_path = 'income_evaluation.csv'
    df = load_data(file_path)

    # İlk EDA'yı gerçekleştir
    initial_eda(df)

    # Veri ön işleme
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Model eğitimi
    model = train_model(X_train, X_test, y_train, y_test)

    # Öznitelik önemi
    feature_scores = feature_importance(model, X_train)
    least_important_feature = feature_scores.idxmin()

    # Modeli değerlendirme
    evaluate_model(model, X_test, y_test)

    # Modeli kaydet
    joblib.dump(model, 'random_forest_model.joblib')

    # Kaydedilen modeli yükle
    loaded_model = joblib.load('random_forest_model.joblib')

    # Yeni veri tanımı
    new_data = {
      "age": 49,
      "workclass": "Private",
      "fnlwgt": 160187,
      "education": "9th",
      "education_num": 5,
      "marital_status": "Married-spouse-absent",
      "occupation": "Other-service",
      "relationship": "Not-in-family",
      "race": "Black",
      "sex": "Female",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 16,
      "native_country": "Jamaica"
    }

    # Yeni veriyi DataFrame'e dönüştür
    new_data_df = pd.DataFrame([new_data])

    # Kodlayıcıyı yükle
    encoder = joblib.load('encoder.joblib')

    # Yüklenen kodlayıcıyı kullanarak yeni veriyi ön işle
    new_data_encoded = encoder.transform(new_data_df)

    # Tahmin yap
    prediction = loaded_model.predict(new_data_encoded)

    print("Predicted income category:", prediction)

if __name__ == "__main__":
    main()
