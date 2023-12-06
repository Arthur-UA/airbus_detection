import json
import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.saving import load_model

"""Gettign the config"""
with open('config.json', 'r') as f:
    CONFIG = json.load(f)


"""Initializing custom loss functions for the model"""
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_model(path=CONFIG['dirs']['model_weights']):
    """Returns the model object"""
    return load_model(path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


def predict(img):
    model = get_model()
    img = tf.image.resize(np.stack([img])/255.0, size=[768, 768])
    result = model.predict(img)[0]

    fig, ax = plt.subplots(1, figsize=(30,30))
    ax.imshow(result)
    ax.axis('off')

    return fig


if __name__ == "__main__":
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type='numpy'),
        outputs=gr.Plot(),
        examples=['1c670582a.jpg', '00b0fa633.jpg', '00b2a7cbd.jpg', '00c81c432.jpg']
    ).launch()