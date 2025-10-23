import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import re
import ast

@st.cache_resource
def carrega_modelo(path_url):
    
    url = converte_url_drive(path_url)
    # url = "https://drive.google.com/uc?id=1R4Oay2HAwm5RajQOwtnV4THKHaM5LJiU"
    gdown.download(url, "modelo_quantizado16bits.tflite")
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte uma imagaem ou clique para selecionar uma", type=["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Imagem Original", use_column_width=True)
        st.success('Imagem carregada com sucesso')


        TENSOR_SIZE = 256
        image = image.resize((TENSOR_SIZE, TENSOR_SIZE))
        st.image(image)
        st.success('Imagem carregada com sucesso')

        image = image.convert('RGB')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def converte_url_drive(url_compartilhamento):
    """
    Extrai o ID do arquivo do link de compartilhamento do Google Drive 
    e o converte para o formato de download direto.
    """
    # Expressão regular para encontrar o ID. Ela busca o ID que vem depois de '/d/' ou 'id='
    match = re.search(r'id=([a-zA-Z0-9_-]+)|/d/([a-zA-Z0-9_-]+)', url_compartilhamento)
    
    if match:
        # match.group(1) captura o ID se for encontrado após 'id='
        # match.group(2) captura o ID se for encontrado após '/d/'
        file_id = match.group(1) if match.group(1) else match.group(2)
        
        # Constrói a URL de download no formato desejado
        url_download_direto = f"https://drive.google.com/uc?id={file_id}"
        return url_download_direto
    else:
        return None


def previsao(interpreter, image, classes):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

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
    # path_url = "https://drive.google.com/file/d/1R4Oay2HAwm5RajQOwtnV4THKHaM5LJiU/view?usp=sharing"
    model_url_input = st.text_input(
        "Cole a URL da imagem (Google Drive ou Link Direto) aqui:",
        value=""
    )
    classes_input_string = st.text_input( # Alterei o nome da variável
        "Lista de classes (Ex: ['BlackMeasles', 'BlackRot', 'HealthGrapes', 'LeafBlight'])",
        value="['BlackMeasles', 'BlackRot', 'HealthGrapes', 'LeafBlight']" # Adicionando um valor padrão para facilitar
    )
    
    interpreter = None
    classes_list = None # Variável para armazenar a lista convertida

    if classes_input_string:
        try:
            # Usamos ast.literal_eval para avaliar a string como um literal Python seguro
            classes_list = ast.literal_eval(classes_input_string)
            if not isinstance(classes_list, list):
                 st.error("Erro na lista de classes: A entrada não é uma lista válida (deve começar e terminar com colchetes []).")
                 classes_list = None
        except Exception:
            st.error("Erro na lista de classes: Verifique a sintaxe (certifique-se de que os itens estão entre aspas simples).")
            classes_list = None
    if model_url_input and classes_list is not None:
        interpreter = carrega_modelo(model_url_input)

    if interpreter is not None:
        st.success("Modelo carregado!")
        st.info("Agora, carregue a imagem da folha de videira para classificação.")
        image = carrega_imagem()
        if image is not None:
            previsao(interpreter, image, classes_list)

    # Carrega imagem

    # Classifica


if __name__=="__main__":
    main()