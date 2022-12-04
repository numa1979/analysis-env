import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use('Agg')

# matplotlib.use("WebAgg")

df_data = pd.read_csv("/workspaces/python-env/work/data/Sensor2.csv", header=None, delimiter="\t")
df_data.shape


plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.xlabel('time')
plt.ylabel('Value')
plt.plot(df_data, color='b')
plt.ylim(-60, 60)
plt.legend()

plt.show()

area_x, area_y = -60, 60

plt.figure(figsize=(15, 5))
plt.xlabel('time')
plt.ylabel('ECG\'s value')
plt.plot(df_data, color='b')
plt.ylim(-60, 60)
x = np.arange(98, 198)
plt.fill_between(x, area_x, area_y, facecolor='r', alpha=.3)
plt.legend()

plt.show()


# 正常データ(0-100)
normal_cycle = df_data[0:100]
# 標準化
x_normal_cycle = (normal_cycle - normal_cycle.mean()) / normal_cycle.std()
# 異常データ(98-198)
abnormal_cycle = df_data[98:198]
# 標準化
x_abnormal_cycle = (abnormal_cycle - normal_cycle.mean()) / normal_cycle.std()


# 正常データplot
plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.xlabel('time')
plt.ylabel('Normal value')
plt.plot(x_normal_cycle, color='b')
plt.ylim(-10, 10)
plt.legend()

plt.show()


# 異常データ
plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.xlabel('time')
plt.ylabel('Abnormal value')
plt.plot(x_abnormal_cycle, color='b')
plt.ylim(-10, 10)
plt.legend()

plt.show()


def create_sequences(df, time_steps):
    x = []
    for i in range(0, len(df) - time_steps + 1):
        x.append(df[i:i + time_steps].to_numpy())
    x_out = np.array(x)
    return x_out


time_steps = 2
x_train = create_sequences(normal_cycle, time_steps)
x_train.shape


model = tf.keras.initializers.Initializer()
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()


# 訓練
history = model.fit(
    x_train,  # 学習データ
    x_train,  # 教師データ
    validation_split=0.1,  # 検証データ比率(学習データの1割を検証データとして使用する)
    epochs=100,  # エポック数
    batch_size=8,  # バッチサイズ
    callbacks=[
        # コールバック指定
        # 5回検証Lossの改善が無かったら学習を打ち切るよう設定
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
threshold = np.max(train_mae_loss)


x_test = create_sequences(x_abnormal_cycle, time_steps)
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))
anomalies = test_mae_loss > threshold


anomalous_data_indices = []
for data_idx in range(time_steps - 1, len(x_abnormal_cycle) - time_steps + 1):
    if np.all(anomalies[data_idx - time_steps + 1: data_idx]):
        anomalous_data_indices.append(data_idx)

df_subset = abnormal_cycle.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_data.plot(legend=False, ax=ax, color="b")
df_subset.plot(legend=False, ax=ax, color="r")

plt.show()
