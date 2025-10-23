import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px


@st.cache_resource
def carrega_modelo():
    url = "https://drive.google.com/uc?id=1it2kDpqrZ6kejH_3JawBCTlSwY0BRz3F"
    # url = "https://drive.google.com/file/d/1it2kDpqrZ6kejH_3JawBCTlSwY0BRz3F/view?usp=drive_link"
    gdown.download(url, "modelo_quantizado16bits.tflite")
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte uma imagaem ou clique para selecionar uma", type=["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem carregada com sucesso')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    
    classes = ['BlackMeasles', 'BlackRot', 'HealthGrapes', 'LeafBlight']

    df = pd.DataFrame()
    df['classes'] = classes

    probabilidades = np.squeeze(output_data)
    df['probabilidades (%)'] = 100 * probabilidades

    fig = px.bar(df, y = 'classes', x = 'probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title = 'Probabilidade de classes de Doenças em Uvas')
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica folha de videira"
    )

    st.write("# Classifica folha s de videira!")

    # Carrega modelo
    interpreter = carrega_modelo()
    image = carrega_imagem()

    if image is not None:
        previsao(interpreter, image)

    # Carrega imagem

    # Classifica


if __name__=="__main__":
    main()