import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('fashion_mnist.keras')

# Definir la interface de usuario
st.title('Clasificador Fashion MNIST')
st.write('Sube una imagen para clasificarla como una categoría de ropa')

uploaded_file = st.file_uploader('Sube una imagen en escala de grises de 28x28 píxeles.')

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L') # Convertir RGB a ByN
    image = image.resize((28, 28)) # Redimensionar a 28x28 píxeles
    img_array = np.array(image) / 255.0 # Normalizar
    img_array = img_array.reshape(1, 28, 28, 1) # El primer 1 indica que sólo hay una imagen, luego las dimensiones, y el último 1 indica que sólo ay un canal de color.

    # Mostramos la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(img_array)
    classes = ['Camiseta/Top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']
    st.write('La predicción es:', classes[np.argmax(prediction)])
