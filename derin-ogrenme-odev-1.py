# Veri Seti (Girdiler ve İkili Sınıflandırma Etiketleri)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

# Ağırlıklar ve Bias (Elle / Rastgele Başlangıç Değerleri)
weights = [0.5, -0.5]
bias = 0.0
learning_rate = 0.1

epoch = 0

# Hata 0 oluncaya kadar çalışacak döngü
while True:
    epoch += 1
    total_error = 0
    dw = [0.0, 0.0]
    db = 0.0
    
    # Batch size = Epoch size (Tüm veri seti tek seferde işleniyor)
    for i in range(len(X)):
        # Net girdi hesaplama: w1*x1 + w2*x2 + bias
        net_input = bias + (X[i][0] * weights[0]) + (X[i][1] * weights[1])
        
        # Aktivasyon fonksiyonu (Eşik değeri: 0)
        prediction = 1 if net_input >= 0 else 0
        
        # Hata hesaplama
        error = y[i] - prediction
        total_error += abs(error)
        
        # Gradyanları biriktir (Batch Güncellemesi için)
        dw[0] += error * X[i][0]
        dw[1] += error * X[i][1]
        db += error
        
    # Eğer toplam hata 0 olduysa döngüyü bitir
    if total_error == 0:
        print(f"Hedefe ulaşıldı! Toplam Hata: 0")
        break
        
    # Ağırlıkları ve bias'ı epoch sonunda güncelle (Batch Gradient Descent)
    weights[0] += learning_rate * dw[0]
    weights[1] += learning_rate * dw[1]
    bias += learning_rate * db
    
    print(f"Epoch {epoch} | Toplam Hata: {total_error} | Ağırlıklar: {weights} | Bias: {bias}")

print(f"\nEğitim Tamamlandı! Toplam Epoch Sayısı: {epoch}")
print(f"Son Ağırlıklar: {weights}, Bias: {bias}")