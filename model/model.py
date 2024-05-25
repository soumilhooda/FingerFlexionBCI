import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, Normalizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.tabular import CopulaGAN

np.random.seed(0)

# Function for calculating AM
def AM(signal):
    cur = 0
    output = []
    for i in range(len(signal)):
        if i and i % 40 == 0:
            output.append(cur)
            cur = 0
        cur += signal[i] ** 2
    output.append(cur)
    return output

dat = [sio.loadmat(f'Data/BCICIV_4_mat/sub{x}_comp.mat', struct_as_record=True) for x in range(1, 4)]
tdat = [sio.loadmat(f'Data/sub{x}_testlabels.mat', struct_as_record=True) for x in range(1, 4)]

train_data = [dat[x]['train_data'] for x in range(3)]
test_data = [dat[x]['test_data'] for x in range(3)]
train_dg = [dat[x]['train_dg'] for x in range(3)]
test_dg = [tdat[x]['test_dg'] for x in range(3)]

train_samples = [train_data[i].shape[0] for i in range(3)]
channels = [train_data[i].shape[1] for i in range(3)]
test_samples = [test_data[i].shape[0] for i in range(3)]
channel_train_data = [np.transpose(train_data[i], (1, 0)) for i in range(3)]
channel_test_data = [np.transpose(test_data[i], (1, 0)) for i in range(3)]
finger_train_data = [np.transpose(train_dg[i], (1, 0)) for i in range(3)]
finger_test_data = [np.transpose(test_dg[i], (1, 0)) for i in range(3)]

sampling_frequency = 1000

def assign_states(finger_data):
    dsamples = len(finger_data[0])
    states = [None] * dsamples
    threshold_1, threshold_2 = 2.0, 1.0
    for i in range(dsamples):
        flex, rest = 0, 0
        for j in range(5):
            if finger_data[j][i] >= threshold_1:
                states[i] = j + 1
                flex += 1
            elif finger_data[j][i] < threshold_2:
                rest += 1
        if states[i] is None:
            if rest:
                states[i] = 0
    return states

merch_train = []
merch_test = []
for i in range(max(channels)):
    merch = []
    for c in range(len(channel_train_data)):
        if i >= channels[c]:
            for _ in range(train_samples[c]):
                merch.append(0)
        else:
            for val in channel_train_data[c][i]:
                merch.append(val)
    merch_train.append(merch)

for i in range(max(channels)):
    merch = []
    for c in range(len(channel_test_data)):
        if i >= channels[c]:
            for _ in range(test_samples[c]):
                merch.append(0)
        else:
            for val in channel_test_data[c][i]:
                merch.append(val)
    merch_test.append(merch)

merf_train = []
for i in range(5):
    merf = []
    for f in finger_train_data:
        for val in f[i]:
            merf.append(val)
    merf_train.append(merf)

merf_test = []
for i in range(5):
    merf = []
    for f in finger_test_data:
        for val in f[i]:
            merf.append(val)
    merf_test.append(merf)

merf_train_ds = [merf_train[i][::40] for i in range(5)]
merf_test_ds = [merf_test[i][::40] for i in range(5)]

merf_train_states = assign_states(merf_train_ds)
merf_test_states = assign_states(merf_test_ds)

train_full_band = [AM(x) for x in merch_train]
test_full_band = [AM(x) for x in merch_test]

trc = np.array(train_full_band)
trf = np.array(merf_train_states)

tec = np.array(test_full_band)
tef = np.array(merf_test_states)

trf = trf.reshape(len(trf), 1)
trf = OneHotEncoder(sparse=False).fit_transform(trf)

tef = tef.reshape(len(tef), 1)
tef = OneHotEncoder(sparse=False).fit_transform(tef)

trc = trc.T
tec = tec.T

trc = Normalizer().fit(trc).transform(trc)
tec = Normalizer().fit(tec).transform(tec)

trc = trc.reshape(len(trc), 1, len(trc[0]))
tec = tec.reshape(len(tec), 1, len(tec[0]))

# Stratified K-Fold Cross Validation and Bootstrapping
n_splits = 5
n_bootstrap_samples = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_x_train = []
all_y_train = []

for train_index, _ in skf.split(trc, trf.argmax(axis=1)):
    X_stratified = trc[train_index]
    y_stratified = trf[train_index]
    for _ in range(n_bootstrap_samples):
        indices = np.random.choice(train_index, len(train_index), replace=True)
        X_bootstrap = trc[indices]
        y_bootstrap = trf[indices]
        all_x_train.append(X_bootstrap)
        all_y_train.append(y_bootstrap)

x_train_extended = np.vstack(all_x_train)
y_train_extended = np.vstack(all_y_train)

# Generating synthetic data using CopulaGAN
data_for_gan = np.hstack((x_train_extended.squeeze(), y_train_extended))

copula_gan = CopulaGAN()
copula_gan.fit(data_for_gan)

num_synthetic_samples = len(x_train_extended)
synthetic_data = copula_gan.sample(num_synthetic_samples)

x_synthetic = synthetic_data[:, :-y_train_extended.shape[1]].reshape(-1, 1, trc.shape[2])
y_synthetic = synthetic_data[:, -y_train_extended.shape[1]:]

x_train_combined = np.vstack((x_train_extended, x_synthetic))
y_train_combined = np.vstack((y_train_extended, y_synthetic))

# Model architecture and training
input = keras.Input(shape=(1, 64))
conv = Conv1D(64, 1, activation='relu')
layer = conv(input)
forward_layer = LSTM(24, activation='relu')
backward_layer = LSTM(24, activation='relu', go_backwards=True)
layer = Bidirectional(forward_layer, backward_layer=backward_layer)(layer)
layer = Dense(12, activation='relu')(layer)
output = Dense(6, activation='softmax')(layer)
model = keras.Model(inputs=input, outputs=output, name="CNN-BiLSTM-BCI")

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

history = model.fit(x_train_combined, y_train_combined, batch_size=128, epochs=100, validation_split=0.05, verbose=1)

# Evaluation
predictions_original = model.predict(tec)
predictions_original = np.argmax(predictions_original, axis=1)
true_labels_original = np.argmax(tef, axis=1)
accuracy_original = accuracy_score(true_labels_original, predictions_original)
confusion_mat_original = confusion_matrix(true_labels_original, predictions_original)

predictions_balanced = model.predict(x_test_balanced)
predictions_balanced = np.argmax(predictions_balanced, axis=1)
true_labels_balanced = np.argmax(y_test_balanced, axis=1)
accuracy_balanced = accuracy_score(true_labels_balanced, predictions_balanced)
confusion_mat_balanced = confusion_matrix(true_labels_balanced, predictions_balanced)

print("Original Test Set Accuracy: ", accuracy_original)
print("Original Test Set Confusion Matrix:\n", confusion_mat_original)

print("Balanced Test Set Accuracy: ", accuracy_balanced)
print("Balanced Test Set Confusion Matrix:\n", confusion_mat_balanced)

plt.figure()
sns.heatmap(confusion_mat_original, annot=True, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Original Test Set Confusion Matrix')
plt.savefig('Confusion_Matrix_Original.png')

plt.figure()
sns.heatmap(confusion_mat_balanced, annot=True, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Balanced Test Set Confusion Matrix')
plt.savefig('Confusion_Matrix_Balanced.png')

plt.figure()
plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Train Accuracy')
plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy_vs_Epoch.png')

plt.figure()
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train Loss')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss_vs_Epoch.png')

