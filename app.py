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
    url = 'https://drive.google.com/uc?id=1ppxxx_uZpXjMD3FpDFMVSNbq3MHfuQWs'

    gdown.download(url,'modelo_quantizado.tflite')
    interpreter = tf.lite.Interpreter(model_path,'modelo_final_transferencia_aprendizado.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_image():
    uploaded = st.file_uploader('Arraste e solte um imagem aqui ou clique para selecionar',type=['png','jpg','jpeg'])
    if uploaded is not None:
        image_data = uploaded.file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem carregada com sucesso!')

        image = np.array(image,dtype=np.float32)
        image = image/255.0
        image = np.expand_dims(image,axis=0)

def previsao(interpreter,image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'],image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot', 'LeafBlight', 'HealthyGrapes']

    df = pd.Dataframe()
    df['classes'] = classes
    df['probabilidades'] = 100 * output_data[0]

    fig = px.bar(df,y='classes',x = 'probabilidades',orientation= 'h',text = 'Probabilidades (%)',
                    title = 'Probabilidade de Classes de Doencas em Uvas')
    
    st.plotly_chart(fig)



def main():

    st.set_page_config(
        page_title="Classifica Folhas de Videira",
    )
    st.write("#Classifica Folhas de Videira!")
    #Carrega modelo
    interpreter = carrega_modelo()
    # Carrega imagem
    image = carrega_image()
    #Classifica

if __name__=="__main__":
    main()