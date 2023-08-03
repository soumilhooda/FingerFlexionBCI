import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

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

trc_finger_nonhotencoded = np.argmax(trf, axis=1)
tec_finger_nonhotencoded = np.argmax(tef, axis=1)

state_indices = [[] for _ in range(6)]
for i, state in enumerate(trc_finger_nonhotencoded):
    state_indices[state].append(i)

min_count = min(len(indices) for indices in state_indices)

balanced_indices = []
for indices in state_indices:
    selected_indices = np.random.choice(indices, min_count, replace=False)
    balanced_indices.extend(selected_indices)

trc_balanced = trc[balanced_indices]
trf_balanced = trf[balanced_indices]

x_train, x_test, y_train, y_test = train_test_split(
    trc_balanced, trf_balanced, test_size=0.2, random_state=42)

x_test_original = tec
y_test_original = tef

x_test_balanced = x_test_original
y_test_balanced = y_test_original

gen_input_shape = 100
gen_output_shape = trc.shape[2]

disc_input_shape = (gen_output_shape,)
disc_output_shape = 1

gen_learning_rate = 0.0002
disc_learning_rate = 0.0002

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

generator = keras.Sequential([
    keras.layers.Dense(256, input_shape=(gen_input_shape,), kernel_initializer=initializer),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dense(512),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dense(1024),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dense(gen_output_shape, activation='tanh')
], name='Generator')

discriminator = keras.Sequential([
    keras.layers.Dense(1024, input_shape=disc_input_shape, kernel_initializer=initializer),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(512),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(disc_output_shape, activation='sigmoid')
], name='Discriminator')

gan = keras.Sequential([generator, discriminator])

discriminator.compile(loss='wasserstein_loss', optimizer=Adam(lr=disc_learning_rate), metrics=['accuracy'])

gan.compile(loss='wasserstein_loss', optimizer=Adam(lr=gen_learning_rate))

epochs = 200
batch_size = 128

discriminator_losses = []
generator_losses = []

for epoch in range(epochs):
    noise = np.random.normal(0, 1, size=[batch_size, 100])
    generated_samples = generator.predict(noise)
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_samples = x_train[idx]
    real_samples = real_samples.reshape(batch_size, 64)
    x = np.concatenate([real_samples, generated_samples], axis=0)
    y_discriminator = np.zeros(2 * batch_size)
    y_discriminator[:batch_size] = 0.9
    discriminator_loss = discriminator.train_on_batch(x, y_discriminator)
    noise = np.random.normal(0, 1, size=[batch_size, 100])
    y_generator = np.ones(batch_size)
    generator_loss = gan.train_on_batch(noise, y_generator)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")
    discriminator_losses.append(discriminator_loss)
    generator_losses.append(generator_loss)

plt.figure()
plt.plot(range(1, epochs + 1), discriminator_losses, label='Discriminator Loss')
plt.plot(range(1, epochs + 1), generator_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('GAN_Losses.png')

num_synthetic_samples = len(x_train)
noise = np.random.normal(0, 1, size=[num_synthetic_samples, 100])
synthetic_samples = generator.predict(noise)
synthetic_samples = np.expand_dims(synthetic_samples, axis=2)
synthetic_samples = synthetic_samples.reshape(synthetic_samples.shape[0], 1, synthetic_samples.shape[1])
x_train_extended = np.concatenate([x_train, synthetic_samples], axis=0)
x_train_extended = x_train_extended.reshape(x_train_extended.shape[0], x_train.shape[1], x_train.shape[2])
y_train_extended = np.concatenate([y_train, y_train[:num_synthetic_samples]])

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
model.compile(loss='wasserstein_loss', optimizer=opt, metrics=['accuracy'])
model.summary()

history = model.fit(x_train_extended, y_train_extended, batch_size=128, epochs=100, validation_split=0.05, verbose=1)

predictions_original = model.predict(x_test_original)
predictions_original = np.argmax(predictions_original, axis=1)
true_labels_original = np.argmax(y_test_original, axis=1)
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

plt.savefig('GAN_Losses.png')
plt.savefig('Confusion_Matrix_Original.png')
plt.savefig('Confusion_Matrix_Balanced.png')
plt.savefig('Accuracy_vs_Epoch.png')
plt.savefig('Loss_vs_Epoch.png')



