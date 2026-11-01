# -*- coding: utf-8 -*-
"""
Bir Regresyon Probleminde Ağ Meta Parametrelerinin Ağ Performansına Etkisinin İncelenmesi

Bu betik, ödevde listelenen altı meta parametre grubunu tek tek (one-factor-at-a-time)
inceler. Her model 300 epoch çalıştırılır ve her deney grubundaki diğer parametreler
sabit tutulur.

ÖNEMLİ NOT:
Ödev metninde hedef fonksiyon "y = 1 + sin(pi * y / 4)" şeklinde yazılmış.
Giriş değişkeni p olarak tanımlandığı için bu betikte muhtemel amaç olan
"y = 1 + sin(pi * p / 4)" kullanılmıştır. Hocanızdan teyit ediniz.

Raporlama:
- MSE, RMSE, R2 ve süre (saniye)
- Her deney grubu için ayrı CSV tablosu ve tüm deneyler için birleşik CSV
- Her deney grubunun en iyi modeli için:
    1) Epoch / MSE (log ölçek)
    2) Giriş / Hedef - Ağ Çıkışı
    3) Epoch / R2

Random seed: 3 (öğrenci numarasına göre belirlenen değer)
"""

# TensorFlow başlamadan önce bazı deterministik çalışma değişkenlerini ayarla.
import os
os.environ.setdefault("PYTHONHASHSEED", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Grafikler ekrana açılmadan PNG olarak kaydedilir.
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. GENEL AYARLAR
# -----------------------------------------------------------------------------
SEED = 3                         # Öğrenci numarasına göre belirlenen random seed
EPOCHS = 300
N_SAMPLES = 201                  # [-2, 2] aralığında örnek sayısı
TEST_EVERY = 5                   # Her 5 noktadan biri test, diğerleri eğitim
BATCH_SIZE = 32
INPUT_MIN = -2.0
INPUT_MAX = 2.0

# Çıktılar bu Python dosyasının bulunduğu klasörün altına yazılır.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sonuclar"
TABLE_DIR = OUTPUT_DIR / "tablolar"
GRAPH_DIR = OUTPUT_DIR / "grafikler"


# -----------------------------------------------------------------------------
# 2. HEDEF FONKSİYON VE VERİ SETİ
# -----------------------------------------------------------------------------
def target_function(p):
    """Ödevdeki giriş değişkeni p için hedef fonksiyon."""
    return 1.0 + np.sin(np.pi * p / 4.0)


def prepare_data():
    """Tüm deneylerde aynı eğitim/test noktalarını kullanır."""
    p = np.linspace(INPUT_MIN, INPUT_MAX, N_SAMPLES, dtype=np.float32)
    y = target_function(p).astype(np.float32)

    # Rastgele bölme yerine aralığa dengeli yayılmış, tekrarlanabilir bölme.
    test_mask = (np.arange(N_SAMPLES) % TEST_EVERY) == 0
    p_train = p[~test_mask].reshape(-1, 1)
    y_train = y[~test_mask].reshape(-1, 1)
    p_test = p[test_mask].reshape(-1, 1)
    y_test = y[test_mask].reshape(-1, 1)

    return p_train, y_train, p_test, y_test


# -----------------------------------------------------------------------------
# 3. AKTİVASYON VE BAŞLANGIÇ DEĞERİ OLUŞTURUCU SEÇİMİ
# -----------------------------------------------------------------------------
ACTIVATION_LABELS = {
    "linear": "Lineer",
    "sigmoid": "Sigmoid",
    "tanh": "Hiperbolik Tanjant",
    "relu": "ReLU",
    "leaky_relu": "LeakyReLU",
    "elu": "ELU",
    "silu": "SiLU",
}


def make_initializer(distribution):
    """Seçilen dağılıma göre yeni bir kernel initializer üretir.

    Normal: N(0, 0.05)
    Uniform: U(-0.05, 0.05)

    Bias değerleri standart uygulama olarak sıfır başlatılır; seçilen dağılım
    Dense katmanlarının ağırlıklarına (kernel) uygulanır.
    """
    if distribution == "normal":
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    if distribution == "uniform":
        return tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
    raise ValueError(f"Bilinmeyen başlangıç dağılımı: {distribution}")


def add_activation(model, activation_name):
    """Seçilen aktivasyon katmanını modele ekler."""
    if activation_name == "leaky_relu":
        # Keras sürümleri arasındaki parametre adı farkını destekle.
        try:
            model.add(tf.keras.layers.LeakyReLU(negative_slope=0.01))
        except TypeError:
            model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    elif activation_name == "elu":
        model.add(tf.keras.layers.ELU(alpha=1.0))
    elif activation_name == "silu":
        # SiLU (x * sigmoid(x)), Swish ile eşdeğerdir.
        model.add(tf.keras.layers.Activation("swish"))
    elif activation_name in ("linear", "sigmoid", "tanh", "relu"):
        model.add(tf.keras.layers.Activation(activation_name))
    else:
        raise ValueError(f"Bilinmeyen aktivasyon fonksiyonu: {activation_name}")


# -----------------------------------------------------------------------------
# 4. MODEL OLUŞTURMA
# -----------------------------------------------------------------------------
def build_model(config):
    model = tf.keras.Sequential(name="regresyon_modeli")
    model.add(tf.keras.layers.Input(shape=(1,), name="giris_p"))

    for _ in range(config["hidden_layers"]):
        model.add(
            tf.keras.layers.Dense(
                units=config["neurons"],
                kernel_initializer=make_initializer(config["initializer"]),
                bias_initializer="zeros",
            )
        )
        add_activation(model, config["activation"])

    # Tek nöronlu lineer çıkış katmanı.
    model.add(
        tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=make_initializer(config["initializer"]),
            bias_initializer="zeros",
            name="lineer_cikis",
        )
    )

    optimizer_name = config["optimizer"]
    learning_rate = config["learning_rate"]

    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Bilinmeyen optimizer: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


# -----------------------------------------------------------------------------
# 5. HER EPOCH SONUNDA TEST R2 DEĞERİNİ HESAPLAMA
# -----------------------------------------------------------------------------
class EpochR2Callback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = np.asarray(y_test).reshape(-1)
        self.r2_values = []
        self.total_sum_squares = float(
            np.sum((self.y_test - np.mean(self.y_test)) ** 2)
        )

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model(self.x_test, training=False).numpy().reshape(-1)
        residual_sum_squares = float(np.sum((self.y_test - predictions) ** 2))
        if self.total_sum_squares == 0:
            r2 = float("nan")
        else:
            r2 = 1.0 - residual_sum_squares / self.total_sum_squares
        self.r2_values.append(r2)


# -----------------------------------------------------------------------------
# 6. TEK BİR MODEL DENEYİNİ ÇALIŞTIRMA
# -----------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator) if denominator else float("nan")
    return mse, rmse, r2


def run_one_experiment(config, p_train, y_train, p_test, y_test):
    """Bir modeli seed=3 ile kurar, 300 epoch eğitir ve test metriklerini döndürür."""
    tf.keras.backend.clear_session()
    gc.collect()

    # Her model için aynı başlangıç random çekirdeğini kullan.
    tf.keras.utils.set_random_seed(SEED)

    start_time = time.perf_counter()
    model = build_model(config)

    r2_callback = EpochR2Callback(p_test, y_test)
    history = model.fit(
        p_train,
        y_train,
        validation_data=(p_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
        callbacks=[r2_callback],
    )

    predictions = model.predict(p_test, verbose=0).reshape(-1)
    mse, rmse, r2 = calculate_metrics(y_test, predictions)
    elapsed_seconds = time.perf_counter() - start_time

    result = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "seconds": float(elapsed_seconds),
        "history": history.history,
        "r2_history": r2_callback.r2_values,
        "model": model,
    }
    return result


# -----------------------------------------------------------------------------
# 7. DENEY GRUPLARI (HER SEFERİNDE TEK PARAMETRE DEĞİŞİR)
# -----------------------------------------------------------------------------
BASE_CONFIG = {
    "activation": "relu",
    "learning_rate": 0.1,
    "optimizer": "sgd",
    "initializer": "normal",
    "hidden_layers": 1,
    "neurons": 2,
}


def make_experiments():
    experiments = []

    # Tablo 1: Aktivasyon fonksiyonu
    values = ["linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu", "silu"]
    experiments.append({
        "key": "01_aktivasyon",
        "title": "Aktivasyon fonksiyonlarının değişimi",
        "parameter": "Aktivasyon",
        "cases": [(v, {**BASE_CONFIG, "activation": v}) for v in values],
    })

    # Tablo 2: Öğrenme oranı
    values = [0.1, 0.01, 0.001]
    experiments.append({
        "key": "02_ogrenme_orani",
        "title": "Öğrenme oranının değişimi",
        "parameter": "Öğrenme oranı",
        "cases": [(v, {**BASE_CONFIG, "learning_rate": v}) for v in values],
    })

    # Tablo 3: Öğrenme oranı ayar yöntemi / optimizer
    values = ["sgd", "rmsprop", "adam"]
    experiments.append({
        "key": "03_optimizasyon_yontemi",
        "title": "Öğrenme oranı ayar yönteminin değişimi",
        "parameter": "Ayar yöntemi",
        "cases": [(v, {**BASE_CONFIG, "optimizer": v}) for v in values],
    })

    # Tablo 4: Ağırlık başlangıç dağılımı
    values = ["normal", "uniform"]
    experiments.append({
        "key": "04_baslangic_dagilimi",
        "title": "Ağırlık başlangıç dağılımının değişimi",
        "parameter": "Başlangıç dağılımı",
        "cases": [(v, {**BASE_CONFIG, "initializer": v}) for v in values],
    })

    # Tablo 5: Gizli katman sayısı
    values = [1, 2, 5, 10, 20]
    experiments.append({
        "key": "05_gizli_katman_sayisi",
        "title": "Gizli katman sayısının değişimi",
        "parameter": "Gizli katman sayısı",
        "cases": [(v, {**BASE_CONFIG, "hidden_layers": v}) for v in values],
    })

    # Tablo 6: Her gizli katmandaki nöron sayısı
    values = [2, 5, 10, 20, 50]
    experiments.append({
        "key": "06_gizli_noron_sayisi",
        "title": "Gizli katmandaki nöron sayısının değişimi",
        "parameter": "Gizli katmandaki nöron sayısı",
        "cases": [(v, {**BASE_CONFIG, "neurons": v}) for v in values],
    })

    return experiments


# -----------------------------------------------------------------------------
# 8. TABLOLARI VE GRAFİKLERİ KAYDETME
# -----------------------------------------------------------------------------
def display_parameter(value, parameter):
    if parameter == "Aktivasyon":
        return ACTIVATION_LABELS.get(value, str(value))
    if parameter == "Ayar yöntemi":
        return {"sgd": "SGD", "rmsprop": "RMSProp", "adam": "ADAM"}.get(value, str(value))
    if parameter == "Başlangıç dağılımı":
        return {"normal": "Normal", "uniform": "Uniform"}.get(value, str(value))
    return str(value)


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(text):
    replacements = {
        "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U",
        "ş": "s", "Ş": "S", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def save_best_graphs(experiment, best_case, p_test, y_test):
    folder = GRAPH_DIR / experiment["key"]
    folder.mkdir(parents=True, exist_ok=True)

    history = best_case["history"]
    epochs = np.arange(1, len(history["loss"]) + 1)
    best_label = display_parameter(best_case["value"], experiment["parameter"])

    # Grafik 1: Epoch / MSE (log)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["loss"], label="Eğitim MSE")
    plt.plot(epochs, history["val_loss"], label="Test MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (logaritmik ölçek)")
    plt.title(f"{experiment['title']} - En iyi değer: {best_label}\nEpoch / MSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "01_epoch_mse_log.png", dpi=160)
    plt.close()

    # Grafik 2: Giriş / Hedef - Ağ çıkışı
    p_plot = np.linspace(INPUT_MIN, INPUT_MAX, 400, dtype=np.float32).reshape(-1, 1)
    y_target = target_function(p_plot.reshape(-1))
    y_output = best_case["model"].predict(p_plot, verbose=0).reshape(-1)

    plt.figure(figsize=(9, 5))
    plt.plot(p_plot.reshape(-1), y_target, label="Hedef")
    plt.plot(p_plot.reshape(-1), y_output, label="Ağ çıkışı")
    plt.scatter(p_test.reshape(-1), y_test.reshape(-1), label="Test örnekleri", s=18)
    plt.xlabel("Giriş (p)")
    plt.ylabel("Çıkış (y)")
    plt.title(f"{experiment['title']} - En iyi değer: {best_label}\nHedef / Ağ Çıkışı")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "02_hedef_ag_cikisi.png", dpi=160)
    plt.close()

    # Grafik 3: Epoch / R2 (test verisi)
    r2_epochs = np.arange(1, len(best_case["r2_history"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(r2_epochs, best_case["r2_history"], label="Test R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title(f"{experiment['title']} - En iyi değer: {best_label}\nEpoch / R²")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "03_epoch_r2.png", dpi=160)
    plt.close()


def run_all_experiments():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Tekrarlanabilirlik ve mümkün olduğunda deterministik TensorFlow işlemleri.
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print("=" * 72)
    print("REGRESYON META PARAMETRE DENEYLERİ BAŞLIYOR")
    print(f"Random seed: {SEED} | Epoch: {EPOCHS} | GPU sayısı: {len(gpus)}")
    print("Her deney grubunda yalnızca ilgili meta parametre değiştirilecektir.")
    print("=" * 72)

    p_train, y_train, p_test, y_test = prepare_data()
    all_rows = []
    best_summary = []

    for experiment in make_experiments():
        print(f"\n{experiment['key']} - {experiment['title']}")
        group_rows = []
        best_case = None

        for index, (value, config) in enumerate(experiment["cases"], start=1):
            shown_value = display_parameter(value, experiment["parameter"])
            print(f"  [{index}/{len(experiment['cases'])}] {experiment['parameter']}: {shown_value} ...", flush=True)

            run_result = run_one_experiment(
                config, p_train, y_train, p_test, y_test
            )

            row = {
                "Deney": experiment["title"],
                "Parametre": experiment["parameter"],
                "Değer": shown_value,
                "MSE": run_result["mse"],
                "RMSE": run_result["rmse"],
                "R2": run_result["r2"],
                "Süre (sn)": run_result["seconds"],
                "Öğrenme oranı": config["learning_rate"],
                "Ayar yöntemi": config["optimizer"],
                "Aktivasyon": ACTIVATION_LABELS[config["activation"]],
                "Başlangıç dağılımı": config["initializer"],
                "Gizli katman": config["hidden_layers"],
                "Nöron/katman": config["neurons"],
                "Epoch": EPOCHS,
                "Random seed": SEED,
            }
            group_rows.append(row)
            all_rows.append(row)

            print(
                f"      MSE={run_result['mse']:.8g} | "
                f"RMSE={run_result['rmse']:.8g} | "
                f"R2={run_result['r2']:.8g} | "
                f"Süre={run_result['seconds']:.2f} sn"
            )

            # En iyi model, test MSE değeri en düşük olan modeldir.
            if math.isfinite(run_result["mse"]):
                if best_case is None or run_result["mse"] < best_case["mse"]:
                    best_case = {
                        "value": value,
                        "config": config.copy(),
                        "mse": run_result["mse"],
                        "rmse": run_result["rmse"],
                        "r2": run_result["r2"],
                        "seconds": run_result["seconds"],
                        "history": run_result["history"],
                        "r2_history": run_result["r2_history"],
                        "model": run_result["model"],
                    }

            # Sonraki deney öncesinde gereksiz model referanslarını bırak.
            del run_result
            gc.collect()

        table_path = TABLE_DIR / f"{experiment['key']}.csv"
        save_csv(table_path, group_rows)

        if best_case is not None:
            save_best_graphs(experiment, best_case, p_test, y_test)
            best_summary.append({
                "Deney": experiment["title"],
                "En iyi parametre": experiment["parameter"],
                "En iyi değer": display_parameter(best_case["value"], experiment["parameter"]),
                "MSE": best_case["mse"],
                "RMSE": best_case["rmse"],
                "R2": best_case["r2"],
                "Süre (sn)": best_case["seconds"],
                "Grafik klasörü": str((GRAPH_DIR / experiment["key"]).relative_to(BASE_DIR)),
            })
            print(
                f"  EN İYİ: {experiment['parameter']} = "
                f"{display_parameter(best_case['value'], experiment['parameter'])} "
                f"(MSE={best_case['mse']:.8g}, R2={best_case['r2']:.8g})"
            )
            del best_case
        else:
            print("  UYARI: Bu grupta sonlu MSE üreten model bulunamadı; grafik oluşturulmadı.")

    save_csv(OUTPUT_DIR / "tum_deneyler.csv", all_rows)
    save_csv(OUTPUT_DIR / "en_iyi_sonuclar.csv", best_summary)

    run_info = {
        "random_seed": SEED,
        "epochs": EPOCHS,
        "number_of_experiments": len(all_rows),
        "samples": N_SAMPLES,
        "training_samples": int(len(p_train)),
        "test_samples": int(len(p_test)),
        "batch_size": BATCH_SIZE,
        "target_function_used": "y = 1 + sin(pi * p / 4)",
        "best_model_selection": "Test MSE en düşük olan model",
        "elapsed_time_note": "Süre; modelin oluşturulması, eğitimi ve test tahminini kapsar.",
    }
    with (OUTPUT_DIR / "deney_bilgileri.json").open("w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 72)
    print(f"TAMAMLANDI. Toplam model deneyi: {len(all_rows)}")
    print(f"Sonuç klasörü: {OUTPUT_DIR}")
    print("Tablolar: sonuclar/tablolar/ ve sonuclar/tum_deneyler.csv")
    print("En iyi sonuçlar: sonuclar/en_iyi_sonuclar.csv")
    print("Grafikler: sonuclar/grafikler/")
    print("=" * 72)


if __name__ == "__main__":
    run_all_experiments()
