import numpy as np

##### BÖLÜM 1 #####

# ndarrays
# 1.
arr_1d = np.array([1, 2, 3, 4, 5])
print('1. ', arr_1d)

# 2.
arr_2d_zeros = np.zeros((3, 4))
print('2. ', arr_2d_zeros)

# 3.
arr_random = np.random.rand(7)
print('3. ', arr_random)

# Vektörler
# 4.
vec_toplama = np.array([1, 2, 3]) + np.array([4, 5, 6])
print('4. ', vec_toplama)

# 5.
vec_iç_çarpım = np.dot([2, 3, 1], [1, 2, 3])
print('5. ', vec_iç_çarpım)

# Matrisler
# 6.
birim_matris = np.eye(3)
print('6. ', birim_matris)

# 7.
matris_A = np.array([[1, 2], [3, 4]])
matris_B = np.array([[5, 6], [7, 8]])
matris_çarpımı = np.dot(matris_A, matris_B)
print('7. ', matris_çarpımı)

# 8.
matris_C = np.array([[1, 2, 3], [4, 5, 6]])
matris_transpoze = matris_C.T
print('8. ', matris_transpoze)

# Nokta Çarpım
# 9.
vec_nokta_çarpım = np.dot([3, 7], [7, 3])
print('9. ', vec_nokta_çarpım)

# 10.
vec_nokta_çarpım2 = np.dot([2, 4], [1, 3])
print('10.', vec_nokta_çarpım2)

# Eleman Bazlı Çarpma
# 11.
arr_eleman_bazlı_çarpma = np.array([2, 3, 4]) * 5
print('11.', arr_eleman_bazlı_çarpma)

# 12.
arr_eleman_bazlı_çarpma_vektörler = np.array([1, 2, 3]) * np.array([4, 5, 6])
print('12.', arr_eleman_bazlı_çarpma_vektörler)

# Matris Çarpımı
# 13.
matris_D = np.array([[1, 2, 3], [4, 5, 6]])
matris_E = np.array([[7, 8], [9, 10], [11, 12]])
matris_çarpım_sonucu = np.dot(matris_D, matris_E)
print('13.', matris_çarpım_sonucu)

# 14.
matris_F = np.array([[2, 3], [4, 5]])
matris_G = np.array([[1, 2], [3, 4]])
matris_çarpım_sonucu_2 = np.dot(matris_F, matris_G)
print('14.', matris_çarpım_sonucu_2)

# ndarrays Oluşturma
# 15.
lineer_aralıklı_dizi = np.linspace(1, 10, 5)
print('15.', lineer_aralıklı_dizi)

# 16.
eşit_aralıklı_dizi = np.linspace(0, 1, 4)
print('16.', eşit_aralıklı_dizi)

# ndarrays Dilimleme
# 17.
ikinci_eleman_H = np.array([10, 20, 30])[1]
print('17.', ikinci_eleman_H)

# 18.
dilimlenmiş_dizi_I = np.array([1, 2, 3, 4, 5])[2:5]
print('18.', dilimlenmiş_dizi_I)

# Matematiksel Fonksiyonlar Örneği
# 19.
ortalama_mutlak_hata = np.mean(np.abs(np.array([5, 7, 9]) - np.array([6, 8, 10])))
print('19.', ortalama_mutlak_hata)

# 20.
karekök_dizisi = np.sqrt(np.array([4, 9, 16]))
print('20.', karekök_dizisi)


##### BÖLÜM 2 #####

import pandas as pd
import matplotlib.pyplot as plt

# Black Friday Veri Seti'ni Yükle
df = pd.read_csv('black_friday_dataset.csv')

# 1. Yükleme ve İlk İnceleme
# a.
print('Veri setinin ilk 5 satırı:')
print(df.head())

# b.
print('\nHer sütunun veri türleri:')
print(df.dtypes)

# c.
print('\nVeri setindeki satır ve sütun sayısı:')
print(df.shape)

# 2. Betimsel İstatistikler
# a.
print("\n'Purchase' sütunu için özet istatistikler:")
print(df['Purchase'].describe())

# b.
print("\n'Age' sütunundaki benzersiz değerler:")
print(df['Age'].unique())

# c.
print('\nHer cinsiyet için ortalama satın alma miktarı:')
print(df.groupby('Gender')['Purchase'].mean())

# 3. Eksik Veri İşleme
# a.
print('\nHer sütündaki eksik değerlerin sayısı:')
print(df.isnull().sum())

# b.
df['Product_Category_2'].fillna(df['Product_Category_2'].mean(), inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].median(), inplace=True)

# 4. Özellik Ölçekleme
# a.
df['Occupation'] = (df['Occupation'] - df['Occupation'].min()) / (df['Occupation'].max() - df['Occupation'].min())

# b.
df['Purchase'] = (df['Purchase'] - df['Purchase'].mean()) / df['Purchase'].std()

# 5. Veriyi Görselleştirme
# a.
plt.hist(df['Age'])
plt.title('Yaş Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Frekans')
plt.show()

# b.
plt.boxplot(df['Purchase'])
plt.title('Satın Alma Miktarı Kutu Grafiği')
plt.ylabel('Satın Alma Miktarı')
plt.show()

# c.
yaş_grubuna_göre_ortalama_satın_alma = df.groupby('Age')['Purchase'].mean()
yaş_grubuna_göre_ortalama_satın_alma.plot(kind='bar', color='skyblue')
plt.title('Yaş Grubuna Göre Ortalama Satın Alma Miktarı')
plt.xlabel('Yaş Grubu')
plt.ylabel('Ortalama Satın Alma Miktarı')
plt.show()

# d.
plt.scatter(df['Occupation'], df['Purchase'], alpha=0.1)
plt.title('Meslek ve Satın Alma Miktarı Scatter Plot')
plt.xlabel('Meslek')
plt.ylabel('Satın Alma Miktarı')
plt.show()
