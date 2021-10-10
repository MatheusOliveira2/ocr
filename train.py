import os
import cv2 # OpenCV ou cv2 para tratamento de imagens;
import numpy as np # Numpy para trabalharmos com matrizes n-dimensionaisz
from tensorflow.keras.models import Sequential # Importando modelo sequencial
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, InputLayer  # Camada de convolução e max pooling
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout # Camada da função de ativação, flatten, entre outros
from tensorflow.keras.layers import BatchNormalization #Camada de normalização de pesos
from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate #Camada de pooling de média global
from tensorflow.keras import backend as K # backend do keras
from tensorflow.keras.optimizers import Adam # optimizador Adam
from tensorflow.keras.preprocessing.image import img_to_array # Função de conversão da imagem para um vetor
from tensorflow.keras.utils import to_categorical # Função utilizada para categorizar listas de treino e teste
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Classe para ajudar na variação de amostras de treinamento
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard # Classe utilizada para acompanhamento durante o treinamento onde definimos os atributos que serão considerados para avaliação
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

def get_data_x_and_y(data_path, width, height):
    
    labels = []
    data = []
    i = 0
    # itera pelo diretório input
    for filename in os.listdir(data_path+'/input'):
        caminho = data_path+'/input/'+filename
        if caminho.endswith(".jpg") or caminho.endswith(".png") or caminho.endswith(".JPEG"):
            
            entrada = cv2.imread(caminho)
            saida = cv2.imread(caminho.replace('input','output'),0)
            
            # converte a imagem para um vetor
            entrada = img_to_array(entrada)
            saida = img_to_array(saida)

            # concatena a imagem a lista de dados que serão utilizados pelo treinamento
            data.append(entrada)
            labels.append(saida)
            
    # Normaliza os dados de treinamento
    X = np.array(data, dtype="float32") / 255.0
    # Categoriza os rotulos
    Y = np.array(labels, dtype="float32") / 255.0
    return (X, Y)

def create_model(input_shape):
    """
    Args:
        input_shape: Uma lista de três valores inteiros que definem a forma de\
                entrada da rede. Exemplo: [100, 100, 1]

    Returns:
        Um modelo sequencial, seguindo a arquitetura lenet
    """
    inputs = Input(shape=input_shape, name = 'image_input')
    X = create_base_model(inputs)
    
    X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
    X = Conv2DTranspose(160, (5, 5), padding="valid")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
    X = Conv2DTranspose(80, (3, 3), padding="valid")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = bloco_inception(X,[10, 20, 10, 20, 10, 10])
    
    X = UpSampling2D(size=(2, 2), interpolation="nearest")(X)
    X = Conv2DTranspose(20, (2, 2), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = concatenate([inputs,X], axis=3)

    X = bloco_inception(X,[2, 4, 2, 4, 2, 2])

    X = Conv2D(4, (2, 2), padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = Conv2D(1, (3, 3), padding="same")(X)
    X = Activation("sigmoid")(X)
    
    model = Model(inputs=inputs, outputs=X, name='minha_rede_inception')

    return model

def create_base_model(inputs):

    base_model = InceptionResNetV2(weights='imagenet', input_shape=(100,100,3), include_top = False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed_5b').output)

    return model(inputs)

def bloco_inception(X,filters):
    '''
    Bloco que surgiu na rede Inception-v1.
    Dilema: quantos filtros é melhor? de qual tamanho? usar pooling? 
    Solução: Não sei... ninguem sabe... deixa a rede descobrir. Faz um tanto de possibilidades e concatena tudo.
    '''
        
    # Quantidad de filtros
    n1, n2, n3, n4, n5, n6 = filters

    conv_1x1 = Conv2D(filters=n1, kernel_size=(1,1), padding='same', activation='relu')(X)
    
    conv_3x3 = Conv2D(filters=n2, kernel_size=(1,1), padding='same', activation='relu')(X)
    conv_3x3 = Conv2D(filters=n3, kernel_size=(3,3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(filters=n4, kernel_size=(1,1), padding='same', activation='relu')(X)
    conv_5x5 = Conv2D(filters=n5, kernel_size=(5,5), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(X)
    pool_proj = Conv2D(filters=n6, kernel_size=(1,1), padding='same', activation='relu')(pool_proj)

    X = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
    
    return X

if __name__ == '__main__':
    train_path = '/Users/CromAI/Documents/ocr/ocr/exemplos/cropagem/train' # Adicione aqui o caminho para chegar no diretório que contém as imagens de treino na sua maquina
    test_path = '/Users/CromAI/Documents/ocr/ocr/exemplos/cropagem/test' # Adicione aqui o caminho para chegar no diretório que contém as imagens de teste na sua maquina
    models_path = "/Users/CromAI/Documents/ocr/ocr/models" # Defina aqui onde serão salvos os modelos na sua maquina
    width = 100 # Tamanho da largura da janela que será utilizada pelo modelo
    height = 100 # Tamanho da altura da janela que será utilizada pelo modelo
    depth = 3 # Profundidade das janelas utilizadas pelo modelo, caso seja RGB use 3, caso escala de cinza 1
    epochs = 20 # Quantidade de épocas (a quantidade de iterações que o modelo realizará durante o treinamento)
    init_lr = 1e-3 # Taxa de aprendizado a ser utilizado pelo optimizador
    batch_size = 32 # Tamanho dos lotes utilizados por cada epoca
    input_shape = (height, width, depth) # entrada do modelo
    save_model = os.path.join(models_path, "segmentador-epoch={epoch:02d}-acc={accuracy:.3f}-val_acc={val_accuracy:.3f}.model")
    os.makedirs(models_path, exist_ok=True)

    (trainX, trainY) = get_data_x_and_y(train_path, width, height)
    (testX, testY) = get_data_x_and_y(test_path, width, height)

    model = create_model(input_shape)
    opt = Adam(learning_rate=init_lr, decay=init_lr / epochs)

    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    model.summary()

    print("\n training network")

    checkpoint1 = ModelCheckpoint(save_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(save_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    tensorboard_dir = os.path.join(models_path, "tensorboard_logs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=tensorboard_dir)

    callbacks_list = [checkpoint1,checkpoint2,tensorboard]
    model.fit(x=trainX,y=trainY,batch_size=batch_size,
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
                            epochs=epochs, verbose=1,callbacks=callbacks_list)