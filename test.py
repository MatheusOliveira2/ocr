import cv2
import numpy as np
import os
import shutil
import sys
import tensorflow as tf

from PIL import Image
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.keras.models import load_model
model_dir = '/Users/CromAI/Documents/ocr/ocr/models/segmentador-epoch=17-acc=0.883-val_acc=0.875.model'

'''converted_model_file = os.path.join(model_dir, "model.pb")
if os.path.exists(converted_model_file):
    pass
else:

    # Load Tensorflow+keras model to be converted
    # TODO: verify if model_path is valid and handle the result
    try:
        loaded = tf.saved_model.load(model_dir)
    except:
        print('Cound not load model')
        sys.exit()

    infer = loaded.signatures['serving_default']
    f = tf.function(infer).get_concrete_function(
        image_input=tf.TensorSpec(shape=(None,100,100,3), dtype=tf.float32))
    f2 = convert_variables_to_constants_v2(f)
    graph_def = f2.graph.as_graph_def()

    # Remove NoOp nodes
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'NoOp':
            del graph_def.node[i]

    for node in graph_def.node:
        for i in reversed(range(len(node.input))):
            if node.input[i][0] == '^':
                del node.input[i]

    # Remove a lot of Identity nodes
    graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def,
                                                            ['image_input'],
                                                            ['Identity'],
                                                            tf.float32.as_datatype_enum)

    with tf.io.gfile.GFile(converted_model_file, 'wb') as f:
        f.write(graph_def.SerializeToString())

model = cv2.dnn.readNetFromTensorflow(model_dir+"/model.pb")'''
model = load_model(model_dir)

data_path = '/Users/CromAI/Documents/ocr/ocr/exemplos/cropagem/test'
for filename in os.listdir(data_path+'/input'):
        caminho = data_path+'/input/'+filename
        if caminho.endswith(".jpg") or caminho.endswith(".png") or caminho.endswith(".JPEG"):
            
            '''pil_image = Image.open(caminho).convert('RGB')
            image = np.array(pil_image)
            image = image[:, :, ::-1].copy()
            image = np.array(image, dtype="float32") / 255.0
            blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(100, 100))
            model.setInput(blob)
            output = model.forward()
            output = output*255
            print(output)'''

            img = cv2.imread(caminho)
            cv2.imshow('input',cv2.resize(img,(250,250)))
            input = (np.expand_dims(img/255,0))

            prediction = model.predict(input)
            output = np.squeeze(prediction,0)
            output=output*255
                 
            cv2.imshow('output',cv2.resize(np.asarray(output[:,:,0],dtype=np.uint8),(250,250)))

            cv2.imwrite(str(caminho.replace('input','predict')), np.asarray(output[:,:,0],dtype=np.uint8))

            cv2.waitKey(0)
