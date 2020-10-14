from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import History 
from keras.optimizers import adam, Nadam
from keras.regularizers import l1
import numpy 

epoch = 500
numpy.random.seed(7)

def autoencoder_model_repo(input_dim,l1_=0.,l2_=0.,l3_=0.):

    encoding_dim = input_dim//2
    input_data = Input(shape=(input_dim,))
    encoded1 = Dense(encoding_dim,activation='relu',kernel_regularizer=l1(l1_))(input_data)
    encoded2 = Dense(encoding_dim,activation='relu',kernel_regularizer=l1(l2_))(encoded1)
    #extraction of encoder
    encoder = Model(input_data, encoded2)
    decoded = Dense(input_dim,activation='sigmoid',kernel_regularizer=l1(l3_))(encoded2)
    autoencoder = Model(input_data, decoded)
    #extraction of decoder
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input,decoder_layer(encoded_input))
    autoencoder.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])

    return autoencoder, encoder, decoder