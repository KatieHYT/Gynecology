from keras.models import Model
from keras.layers import Input,Conv1D, Dense, MaxPool1D, Activation, AvgPool1D,GlobalAveragePooling1D
from keras.layers import Flatten, Add, Concatenate, Dropout, BatchNormalization
from keras.regularizers import l2

def ResidualBlock(filters,kernel_size,strides,pool_size,inputs, l_2=0.0, activation='relu', kernel_initializer='he_normal'):
    path1 = MaxPool1D(pool_size=pool_size, padding = 'same', strides = strides)(inputs)
    
    path2 = BatchNormalization()(inputs)
    path2 = Activation(activation=activation)(path2)
    path2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                   kernel_regularizer = l2(l_2),
                   kernel_initializer = kernel_initializer)(path2)
    path2 = BatchNormalization()(path2)
    path2 = Activation(activation=activation)(path2)
    path2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same', 
                   kernel_regularizer = l2(l_2),
                   kernel_initializer = kernel_initializer)(path2)
    path2 = Add()([path2, path1])
    return path2

def build_model(length=600, n_channel=2, n_classes=2, filters=64, kernel_size=3, layers = 10,
                activation='relu',kernel_initializer = 'he_normal', l_2=0.0):    
    sig_inp =  Input(shape=(length, n_channel))  
    inp = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding="same", 
                 kernel_regularizer=l2(l_2))(sig_inp)
    inp = BatchNormalization()(inp)
    inp = Activation(activation=activation)(inp)
    inp_max = MaxPool1D(pool_size=2)(inp)

    l1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding="same",
                kernel_regularizer=l2(l_2))(inp)
    l1 = BatchNormalization()(l1)
    l1 = Activation(activation=activation)(l1)
    l1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding="same",
                kernel_regularizer=l2(l_2))(l1)

    new_inp = Add()([l1,inp_max])

    for i in range(layers):
    # every alternate residual block subsample its input by a factor of 2
        if i % 2 == 1:
            pool_size = 2
            strides = 2
        else:
            pool_size = 1
            strides = 1
        # incremented filters    
        if i % 4 == 3:
            filters = 64*int(i//4 + 2)
            new_inp = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same',
                             kernel_regularizer=l2(l_2),
                             kernel_initializer = kernel_initializer)(new_inp)
        new_inp = ResidualBlock(filters,kernel_size,strides,pool_size,new_inp, l_2=l_2)

    list_c = []
    
    new_inp_man = GlobalAveragePooling1D()(new_inp)
    new_inp_man = BatchNormalization()(new_inp_man)
    list_c.append(new_inp_man)
    new_inp_man = Dense(128, kernel_regularizer=l2(l_2))(new_inp_man) 
    new_inp_man = BatchNormalization()(new_inp_man)
    new_inp_man = Activation(activation=activation)(new_inp_man)
    out_man = Dense(3, activation='softmax', kernel_regularizer=l2(l_2), name='man')(new_inp_man)
    list_c.append(out_man)
    
    new_inp_ua = GlobalAveragePooling1D()(new_inp)
    new_inp_ua = BatchNormalization()(new_inp_ua)
    new_inp_ua = Dense(128, kernel_regularizer=l2(l_2))(new_inp_ua) 
    new_inp_ua = BatchNormalization()(new_inp_ua)
    new_inp_ua = Activation(activation=activation)(new_inp_ua)
    out_ua  = Dense(2, activation='softmax', kernel_regularizer=l2(l_2), name='ua' )(new_inp_ua)
    list_c.append(out_ua)
    
    new_inp_var = GlobalAveragePooling1D()(new_inp)
    new_inp_var = BatchNormalization()(new_inp_var)
    new_inp_var = Dense(128, kernel_regularizer=l2(l_2))(new_inp_var) 
    new_inp_var = BatchNormalization()(new_inp_var)
    new_inp_var = Activation(activation=activation)(new_inp_var)
    out_var = Dense(2, activation='softmax', kernel_regularizer=l2(l_2), name='var')(new_inp_var)
    list_c.append(out_var)
    
    
    new_inp_dec = GlobalAveragePooling1D()(new_inp)
    new_inp_dec = BatchNormalization()(new_inp_dec)
    new_inp_dec = Dense(128, kernel_regularizer=l2(l_2))(new_inp_dec) 
    new_inp_dec = BatchNormalization()(new_inp_dec)
    new_inp_dec = Activation(activation=activation)(new_inp_dec)
    out_dec = Dense(4, activation='softmax', kernel_regularizer=l2(l_2), name='dec')(new_inp_dec)
    list_c.append(out_dec)   
    
    new_inp_fhb = GlobalAveragePooling1D()(new_inp)
    new_inp_fhb = BatchNormalization()(new_inp_fhb)
    new_inp_fhb = Dense(128, kernel_regularizer=l2(l_2))(new_inp_fhb) 
    new_inp_fhb = BatchNormalization()(new_inp_fhb)
    new_inp_fhb = Activation(activation=activation)(new_inp_fhb)
    out_fhb = Dense(4, activation='softmax', kernel_regularizer=l2(l_2), name='fhb')(new_inp_fhb)
    list_c.append(out_fhb)    
 
    man_concat = Concatenate()(list_c)
    man_concat = BatchNormalization(name='man_concat_BN1')(man_concat)
    man_concat = Dense(256, kernel_regularizer=l2(l_2))(man_concat)
    man_concat = BatchNormalization(name='man_concat_BN2')(man_concat)
    man_concat = Dense(128, kernel_regularizer=l2(l_2))(man_concat)
    man_concat = BatchNormalization(name='man_concat_BN3')(man_concat)
    man_concat = Activation(activation=activation)(man_concat)
    out_man_concat = Dense(3, activation='softmax', kernel_regularizer=l2(l_2), name='man_concat')(man_concat)
    
    
    
    model = Model(inputs=[sig_inp], 
                  outputs=[out_man, out_ua, out_var, out_dec, out_fhb, out_man_concat])
    return model

