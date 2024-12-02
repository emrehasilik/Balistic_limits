# Gerekli kütüphaneleri içe aktarın
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

# Uyarıları bastır
warnings.filterwarnings('ignore')

def train_model(data_path):
    # 1. Veri setini oku
    data = pd.read_excel(data_path)
    data.columns = data.columns.str.strip()
    
    # 2. Girdileri ve hedef değişkeni ayırın
    X = data[['Ball Diameter', 'Composite Matrix Thickness']].values
    y = data['Ballistic Limit'].values
    
    # 3. Özellikleri ölçeklendirme
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 4. Modeli oluşturun
    model = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
    
    # 5. Çapraz doğrulama ile hata payını değerlendirin
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mape_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_percentage_error')
    mean_mape = -np.mean(mape_scores) * 100  # Yüzdeye çevirme
    
    print(f"Ortalama MAPE (Mean Absolute Percentage Error): {mean_mape:.2f}%")
    
    # 6. Tahmin hatalarının standart sapmasını hesaplayın (Tahmin aralığı için)
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)
    mse = mean_squared_error(y, y_pred_cv)
    std_error = np.sqrt(mse)
    
    # 7. Modeli tüm veri ile eğitin
    model.fit(X_scaled, y)
    
    return model, scaler_X, mean_mape, std_error, X, y

def create_gui():
    # Modeli eğitin
    data_path = 'veri_seti.xlsx'
    model, scaler_X, mean_mape, std_error, X, y = train_model(data_path)
    
    # GUI oluştur
    root = tk.Tk()
    root.title("Ballistic Limit Tahmin Uygulaması - SVR")
    
    # Giriş alanları ve etiketler
    tk.Label(root, text="Ball Diameter (mm):").grid(row=0, column=0, padx=10, pady=10)
    entry_diameter = tk.Entry(root)
    entry_diameter.grid(row=0, column=1, padx=10, pady=10)
    
    tk.Label(root, text="Composite Matrix Thickness (mm):").grid(row=1, column=0, padx=10, pady=10)
    entry_thickness = tk.Entry(root)
    entry_thickness.grid(row=1, column=1, padx=10, pady=10)
    
    # Sonuç etiketi
    label_result = tk.Label(root, text="")
    label_result.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    
    # Tahmin ve görselleştirme fonksiyonu
    def predict_and_plot():
        try:
            ball_diameter = float(entry_diameter.get())
            composite_thickness = float(entry_thickness.get())
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli sayısal değerler girin.")
            return
        
        # Yeni veriyi hazırlayın
        new_data = np.array([[ball_diameter, composite_thickness]])
        new_data_scaled = scaler_X.transform(new_data)
        
        # Tahmin yapın
        prediction = model.predict(new_data_scaled)[0]
        
        # Hata payını hesaplayın (Tahmin aralığı)
        fixed_error=1.5
        lower_bound = prediction - fixed_error
        upper_bound = prediction + fixed_error
        
        # Sonucu göster
        result_text = f"Tahmin Edilen Ballistic Limit: {prediction:.2f} m/s\n"
        result_text += f"Aralık: [{lower_bound:.2f}, {upper_bound:.2f}] m/s"
        label_result.config(text=result_text)
        
        # Grafik oluştur ve göster
        fig, ax = plt.subplots(figsize=(6,4))
        
        # Mevcut verileri çiz
        ax.scatter(X[:,1], y, color='blue', label='Gerçek Değerler')
        
        # Tahmin eğrisi için değerler
        thickness_range = np.linspace(X[:,1].min(), X[:,1].max(), 100)
        diameter_array = np.full(shape=thickness_range.shape, fill_value=ball_diameter)
        plot_data = np.column_stack((diameter_array, thickness_range))
        plot_data_scaled = scaler_X.transform(plot_data)
        predictions = model.predict(plot_data_scaled)
        
        # Tahmin eğrisini çiz
        ax.plot(thickness_range, predictions, color='red', label='Tahmin Eğrisi')
        
        # Yeni nokta
        ax.scatter(composite_thickness, prediction, color='green', label='Yeni Tahmin', zorder=5)
        
        ax.set_xlabel('Composite Matrix Thickness (mm)')
        ax.set_ylabel('Ballistic Limit (m/s)')
        ax.set_title(f'Ball Diameter = {ball_diameter} mm için Tahmin - SVR')
        
        ax.legend()
        ax.grid(True)
        
        # Grafiği tkinter içinde göster
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=0, columnspan=2)
    
    # Tahmin butonu
    btn_predict = tk.Button(root, text="Tahmin Yap", command=predict_and_plot)
    btn_predict.grid(row=2, column=1, padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
