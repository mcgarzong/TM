jimport streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Carga del modelo y etiquetas
model = load_model('bienymal.h5')

# Cargar etiquetas desde el archivo de texto
with open("texto.txt", "r") as file:
    labels = [line.strip() for line in file.readlines()]

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- INTERFAZ PRINCIPAL ---

st.title("🪞 El Espejo de las Decisiones")
st.caption("Tu espejo mágico interpreta tus gestos y revela si la energía que proyectas es positiva o negativa. ✨")

image = Image.open('espejo.jpeg')
st.image(image, width=350, caption="Deja que el espejo lea tu energía...")

with st.sidebar:
    st.subheader("🔮 Instrucciones")
    st.write("1️⃣ Coloca tu mano frente a la cámara. \n\n"
             "2️⃣ Haz el gesto de **pulgar arriba 👍** o **pulgar abajo 👎**. \n\n"
             "3️⃣ Espera el veredicto del espejo mágico.")
    st.info("Recuerda: cada gesto cambia la energía del momento 💫")

img_file_buffer = st.camera_input("✨ Muestra tu gesto frente al espejo")

if img_file_buffer is not None:
    # Convertir la imagen capturada
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)

    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar la predicción
    prediction = model.predict(data)
    index = np.argmax(prediction)
    gesture = labels[index]
    confidence = prediction[0][index]

    # Mostrar resultado
    if gesture.lower() == "bien":
        st.success(f"🌟 Energía positiva detectada ({confidence:.2f})")
        st.markdown("Tu espejo refleja **buenas vibras** y claridad interior. ✨")
    elif gesture.lower() == "mal":
        st.error(f"🌫️ Energía negativa detectada ({confidence:.2f})")
        st.markdown("El ambiente se siente denso... Respira y vuelve a intentarlo 🕯️")
    else:
        st.warning("🤔 Gesto no reconocido. Asegúrate de mostrar bien el pulgar.")

    st.button("🔁 Reiniciar lectura")
