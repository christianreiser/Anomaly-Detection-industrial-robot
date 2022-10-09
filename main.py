import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
from keras import layers, losses
from keras.models import Model


data = pd.read_parquet('20221007_CC_AD_DATA.parquet', engine='fastparquet')

# remove a6
data = data.drop([col for col in data.columns if col.startswith('a6')], axis=1)


# train test split: 2017-09-11 are training data, the rest are testing data
train = data[data.loc[:,'robtime'] < '2017-09-12']
test = data[data.loc[:,'robtime'] < '2017-09-12']

# drop column robtime as not useful for anomaly detection
train = train.drop(['robtime'], axis=1)
test = test.drop(['robtime'], axis=1)

# drop a4_torque_diff
test = test.drop(['a4_torque_diff'], axis=1)
train = train.drop(['a4_torque_diff'], axis=1)

# normalize data
train_norm = (train - train.mean()) / train.std()
test_norm = (test - test.mean()) / test.std()



# Build the model

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(6, activation="relu"),
      layers.Dense(1, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(6, activation="relu"),
      layers.Dense(24, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(train_norm, train_norm,
          epochs=10,
          batch_size=512,
          validation_data=(test_norm, test_norm),
          shuffle=True)




reconstructions = autoencoder.predict(train_norm)
train_loss = tf.keras.losses.mae(reconstructions, train_norm)
plt.hist(train_loss.numpy(), bins=50)
threshold = 1.0
# vertical line at the threshold
plt.vlines(threshold, ymin=0, ymax=100, colors="r", zorder=100, label='Threshold')
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()


reconstructions = autoencoder.predict(test_norm)
loss = tf.keras.losses.mae(reconstructions, test_norm)
prediction = tf.math.less(loss, threshold)
# print number of true in prediction
print(prediction.numpy().sum())
print('fraq of anomalies in test data: {}'.format(prediction.numpy().sum()/len(test)))


isofor_score = [-1,-2,1,2]
# create new variable isofor_score_binary with True for anomalies and False for normal
isofor_score_binary = []
for i in isofor_score:
    if i <0:
        isofor_score_binary.append(True)
    else:
        isofor_score_binary.append(False)

# plot isofor_score_binary and prediction as subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(isofor_score_binary)
ax1.set_title('isofor_score_binary')
ax2.plot(prediction)
ax2.set_title('prediction')
plt.xlabel("Time")
plt.show()

