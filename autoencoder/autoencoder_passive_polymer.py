import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import numpy as np
import tensorflow as tf

import os
import random
from glob import glob
import shutil
import pickle
from tqdm import tqdm

seed_value= 12345
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


if(len(glob("save_model_polymer"))):
       shutil.rmtree("save_model_polymer")
os.mkdir("save_model_polymer")


if(len(glob("save_model_polymer_best"))):
       shutil.rmtree("save_model_polymer_best")
os.mkdir("save_model_polymer_best")

if(len(glob("training_loss_polymer"))):
       shutil.rmtree("training_loss_polymer")
os.mkdir("training_loss_polymer")


if(len(glob("output_data_polymer"))):
       shutil.rmtree("output_data_polymer")
os.mkdir("output_data_polymer")


if(len(glob("latent_data_polymer"))):
       shutil.rmtree("latent_data_polymer")
os.mkdir("latent_data_polymer")


if(len(glob("model_loss_polymer"))):
       shutil.rmtree("model_loss_polymer")
os.mkdir("model_loss_polymer")


dist_data = pickle.load(open(f"../../../polymer_simulation/dist_matrix_for_32_bead_polymer_skip_2.pkl", "rb"))


req_data = dist_data[:, ::4, ::4]

upper_triangular_elements_list = []

pbar = tqdm(total = int(len(req_data)), desc = "Progress")
for sample in req_data:
    upper_triangular_elements = sample[np.triu_indices(sample.shape[0], k=1)]
    # upper_triangular_elements = upper_triangular_elements/upper_triangular_elements.max()
    upper_triangular_elements_list.append(upper_triangular_elements)
    pbar.update()


final_data = np.array(upper_triangular_elements_list)
pbar.close()
print(final_data.shape)

##with flattening
train_data = final_data
samples = train_data.shape[0]
flatten = train_data.shape[1]

input_data = tf.keras.utils.normalize(train_data, axis=1)
print(input_data.shape)



dimension = []
fraction = []
rec_loss = []
for f in range(1, 6, 1):
    input_dim = flatten
    hidden_dim1 = 12
    latent_dim = int(f)
    output_dim = flatten

    input_layer = tf.keras.Input(shape=(input_dim,), name = "INPUT")
    hidden_layer1 = tf.keras.layers.Dense(hidden_dim1, activation='elu') (input_layer)
    encoded = tf.keras.layers.Dense(latent_dim, activation='elu', name = "BOTTLE") (hidden_layer1)
    decoded1 = tf.keras.layers.Dense(hidden_dim1, activation='elu')(encoded)
    output_layer = tf.keras.layers.Dense(output_dim, activation='elu', name = "OUTPUT")(decoded1)
    AE = tf.keras.models.Model(input_layer, output_layer)
    print(AE.summary())

    # optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    AE.compile(optimizer=optimizer, loss='mse')
    batch_size = 100
    epochs = 500
    #we are not using best model, we are using the model parameter for last Epochs 
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"save_model_polymer_best/save_model_polymer_latent_dim_{latent_dim}_best.keras", monitor='val_accuracy', save_best_only=True, mode='max')
    history = AE.fit(input_data, input_data, epochs=epochs, shuffle=True, batch_size=batch_size,validation_split=0.1, callbacks=[checkpoint])

    ##saving the model parameter
    AE.save(f"save_model_polymer/save_model_polymer_latent_dim_{latent_dim}.keras")

    ##calculating the training loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_loss = np.array(val_loss)
    train_loss = np.array(train_loss)
    num_epochs = [int(value) for value in np.linspace(1, epochs, epochs) if value.is_integer()]
    num_epochs = np.array(num_epochs)
    np.savetxt(f"training_loss_polymer/training_loss_polymer_latent_dim_{latent_dim}.txt", np.array([num_epochs, train_loss, val_loss]).T, delimiter = "\t", fmt = "%0.3e")

    encoded_data = AE.predict(input_data)
    print(encoded_data.shape)
    np.savetxt(f"output_data_polymer/matrix_{latent_dim}.mat", encoded_data, fmt = "%0.3e")

    reconstruction_loss= AE.evaluate(input_data, input_data)
    print("reconstruction_loss = ", reconstruction_loss)
    rec_loss.append(reconstruction_loss)

    # Calculate the Fraction of Variance Explained (FVE)
    total_variance = np.sum((input_data - np.mean(input_data, axis = 0))**2)
    reconstruction_error = np.sum((input_data - encoded_data)**2)
    fve = 1 - (reconstruction_error / total_variance)
    print("Fraction of Variance Explained (FVE):", fve, latent_dim)
    dimension.append(latent_dim)
    fraction.append(fve)

    ##data from latent dimension
    encoder = tf.keras.models.Model(AE.input, AE.get_layer("BOTTLE").output) 
    latent_data = encoder.predict(input_data)
    np.savetxt(f"latent_data_polymer/latent_data_polymer_latent_dim_{latent_dim}.txt", latent_data, delimiter = "\t", fmt = "%0.3e")



dimension = np.array(dimension)
fraction = np.array(fraction)
np.savetxt("model_loss_polymer/fraction_of_variance_polymer.txt", np.array([dimension, fraction]).T, fmt = "%0.3e", delimiter = "\t")
np.savetxt("model_loss_polymer/reconstruction_loss_polymer.txt", np.array([dimension, rec_loss]).T, fmt = "%0.3e", delimiter = "\t")
