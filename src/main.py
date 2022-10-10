import pickle
import pandas as pd
import shap
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses
from keras.models import Model
from sklearn.ensemble import IsolationForest
from flask import Flask

# app = Flask('FinTechExplained WebServer')
#
# @app.route('/')
# def get_data():
#     return [1,2,3]


def isolation_forest(train, test):
    clf1 = IsolationForest(random_state=0).fit(train)
    pred = clf1.predict(test)
    isofor_score = clf1.decision_function(test)
    neg_value_indices = np.where(isofor_score < 0)
    frac_anomalies_isolation_forst = len(neg_value_indices[0]) / len(test)

    # Tree Explainer
    # create an explainer object
    explainer = shap.TreeExplainer(clf1, data=train)
    plt.close()

    # calculate shap values
    # shapvs = explainer.shap_values(train, check_additivity=False)
    # load
    with open('./src/shapvs.pickle', 'rb') as f:
        shapvs = pickle.load(f)

    # todo plots
    # shap.summary_plot(shapvs, train, plot_type="bar", color_bar=True)
    # plt.savefig('./plots/isolation_forest_shap_summary_plot.png')
    # plt.close()

    # shap.summary_plot(shapvs, train, color_bar=True, feature_names=train.columns)
    # plt.savefig('./plots/isolation_forest_shap_summary_plot2.png')
    # plt.close()
    return pred, frac_anomalies_isolation_forst


def autoencoder(train, test, frac_anomalies_isolation_forst):
    # normalize data
    train_norm = (train - train.mean()) / train.std()
    test_norm = (test - test.mean()) / test.std()

    # drop a4_torque_diff
    test_norm = test_norm.drop(['a4_torque_diff'], axis=1)
    train_norm = train_norm.drop(['a4_torque_diff'], axis=1)

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

    # train the model
    history = autoencoder.fit(train_norm, train_norm,
                              epochs=20,
                              batch_size=512,
                              validation_data=(test_norm, test_norm),
                              shuffle=True)

    # plot training validation loss over time
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig('./plots/autoencoder_train_loss_history.png')
    plt.close()

    # predict training data
    reconstructions_train = autoencoder.predict(train_norm)
    train_loss = tf.keras.losses.mae(reconstructions_train, train_norm)
    plt.scatter(range(len(train_loss)), train_loss, c='b', s=3, label='train loss')
    plt.xlabel("No of examples")
    plt.ylabel("Reconstruction loss")
    plt.savefig('./plots/autoencoder_rec_train_loss.png')
    plt.close()

    # get threshold so fraction between iso forest and autoencoder is the same
    threshold_autoencoder = np.percentile(train_loss[None, :], (1 - frac_anomalies_isolation_forst) * 100)

    plt.hist(train_loss[None, :], bins=50)
    plt.vlines(1.5, ymin=0, ymax=35000, colors="r", zorder=100, label='Threshold1')
    plt.vlines(0.5, ymin=0, ymax=35000, colors="r", zorder=100, label='Threshold2')
    plt.vlines(threshold_autoencoder, ymin=0, ymax=35000, colors="g", zorder=100, label='threshold_autoencoder')
    plt.xlabel("Train loss")
    plt.ylabel("No of samples")
    plt.savefig('./plots/autoencoder_rec_train_loss_hist.png')
    plt.close()

    # predict test data
    reconstructions_test = autoencoder.predict(test_norm)
    loss_test = tf.keras.losses.mae(reconstructions_test, test_norm)
    prediction_autoencoder_test = tf.math.greater(loss_test, threshold_autoencoder)
    return prediction_autoencoder_test



def main():
    # load data
    data = pd.read_parquet('./src/20221007_CC_AD_DATA.parquet', engine='fastparquet')

    # remove columns without info
    # remove a6
    data = data.drop([col for col in data.columns if col.startswith('a6')], axis=1)

    # train test split: 2017-09-11 are training data, the rest are testing data
    train = data[data.loc[:, 'robtime'] < '2017-09-12']
    test = data[data.loc[:, 'robtime'] > '2017-09-12']

    # drop column robtime as not useful for anomaly detection
    train = train.drop(['robtime'], axis=1)
    test = test.drop(['robtime'], axis=1)

    prediction_isolation_forest_test, frac_anomalies_isolation_forst = isolation_forest(train, test)
    print('prediction_isolation_forest_test:', prediction_isolation_forest_test)
    prediction_autoencoder_test = autoencoder(train, test, frac_anomalies_isolation_forst)
    print('prediction_autoencoder_test:', prediction_autoencoder_test)

main()