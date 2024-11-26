import streamlit as st
from PIL import Image, ImageOps
import pdfplumber
import pytesseract
import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
import tempfile
import os

st.set_page_config(page_title= "SIIA FACTURAS",layout="wide")

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

NN=load_model('base_final.keras')

st.markdown("""
  <style>
  div.stSpinner>div {
    text-align:center;
    align-items: center;
    justify-content: center;
  }
  </style>""", unsafe_allow_html=True)

st.header("SIIA AVALUOS")
st.subheader("Validación de facturas", divider='orange')
col1, col2, col3, col4= st.columns([2,3,3,1])

img=Image.open('siia.jpg')
col1.image(img, use_column_width=True )

excel=col3.file_uploader("Arrastra o sube la hoja de cálculo.", type=['xls'])

facturas = col2.file_uploader("Arrastra o sube aquí tus archivos.", accept_multiple_files=True, type=['pdf'])

_, col23, _ = st.columns([2, 6, 1])

#col23.write("En caso de tener facturas de Constructodo, favor de subirla en este espacio.",accept_multiple_files=True, type=['pdf'])
#facturas_constru=col23.file_uploader("Arrastra o sube las facturas de la empresa Constructodo.")

facturas_borrosas=col23.text_area("Si considera necesario, ingresa los valores de facturas borrosas separados por comas")
        

if facturas_borrosas:
    try:
        facturas_borrosas = np.array([float(num.strip()) for num in facturas_borrosas.split(",")])
    except ValueError:
        facturas_borrosas=[]
        col23.warning("Favor de ingresar solo números separados por comas.")

B=col23.button("Validar facturas", help="Verifica y valida los totales de las facturas.")

#if(facturas_constru):
#    facturas.append(1)

if(B):
    with st.spinner("Procesando..."):
        if (excel and facturas):
            totales_pred=[]
            for P in facturas:
                with pdfplumber.open(P) as pdf:
                    pagina = pdf.pages[0]
                    im = pagina.to_image(resolution=250).original
                    im=ImageOps.invert(im)
                        
                    largo=im.size[0]
                    validar_total_encontrado=0

                    datos = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)

                    for i, texto in enumerate(datos["text"]):
                            if "total" in texto.lower():
                                x, y, w, h = datos["left"][i], datos["top"][i], datos["width"][i], datos["height"][i]
                                #print(f"Palabra 'Total' encontrada en: ({x}, {y}, {w}, {h})")

                                seccion = im.crop((x+2*w, y-8, largo, y + h + 10))
                                validar_total_encontrado=1

                    if (validar_total_encontrado==0):
                        col23.warning(f"No fue posible encontrar el total en la factura {P.name}.")    
                    else:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                            seccion.save(tmpfile.name)
                            temp_file_path = tmpfile.name

                        image = cv2.imread(temp_file_path)

                        os.remove(temp_file_path)

                        image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

                        imagen_nitida = cv2.filter2D(image_g, -1, k)

                        _, blanco_negro = cv2.threshold(imagen_nitida, 150, 255, cv2.THRESH_BINARY)

                        if cv2.countNonZero(blanco_negro) > (blanco_negro.size // 2):
                            blanco_negro = cv2.bitwise_not(blanco_negro)

                        contornos, _ = cv2.findContours(blanco_negro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        contornos_digitos = [c for c in contornos if cv2.contourArea(c) > 60 and cv2.contourArea(c) < 470]

                        contornos_digitos = sorted(contornos_digitos, key=lambda x: cv2.boundingRect(x)[0])

                        digitos_imagenes = []

                        for c in contornos_digitos:
                            x, y, w, h = cv2.boundingRect(c)
                            digito = blanco_negro[y-3:y+h+3, x-3:x+w+3]
                            digitos_imagenes.append(digito)
                            col23.image(digito)

                        digitos_imagenes = [cv2.resize(digito, (32, 32)) for digito in digitos_imagenes]

                        digitos_erosion = []
                        for i, digit in enumerate(digitos_imagenes):
                            kernel = np.ones((2,2))
                            e_digit = cv2.erode(digit, kernel, iterations=1)
                            digitos_erosion.append(e_digit)

                        if digitos_erosion:
                            digitos_cnn=np.stack(digitos_erosion)

                            digitos_cnn=digitos_cnn.reshape(digitos_cnn.shape[0],32,32,1)

                            cantidad=[]
                            for i in range(digitos_cnn.shape[0]):
                                image = digitos_cnn[i]
                                image = image.reshape(1, 32, 32, 1)

                                prediction = NN.predict(image)
                                digit = np.argmax(prediction)
                                cantidad.append(digit)

                            cantidad=np.array(cantidad)
                            cantidad=np.flip(cantidad)
                            T=0
                            p=-2

                            for i in cantidad:
                                if(i!=10):
                                    T+=i*(10**(p))
                                    p+=1
                            totales_pred.append(round(T, ndigits=2))
                        else:
                            col23.warning(f"No fue posible encontrar el total en la factura {P.name}.")

            #Hasta aquí llega el for de las facturas

            excel=pd.read_excel(excel)
            totales_pred=np.array(totales_pred)
            data=excel[excel['proveedor'].isnull()]

            total_facturas=data['vencido'].values
            total_facturas=total_facturas[np.nonzero(total_facturas)]
        
            validacion=np.full(len(total_facturas), "Total Incorrecto")

            if facturas_borrosas:
                totales_pred=np.concatenate((totales_pred, facturas_borrosas))

            if(len(total_facturas)!=len(totales_pred)):
                totales_pred = np.concatenate([totales_pred, np.zeros(len(total_facturas) - len(totales_pred))])
            
            indices = np.isin(total_facturas, totales_pred)
            validacion[indices] = "Correcto"
            
            indices_2=~np.isin(totales_pred, total_facturas)
            totales_revisados=totales_pred[indices_2]
            totales_revisados = np.concatenate([totales_revisados, np.zeros(len(total_facturas) - len(totales_revisados))])
            #for i in totales_pred:
            #    for j in total_facturas:
            #        if(i==j):
            #            index = total_facturas.index(j)
            #            st.write(index)
            #            validacion[index] = "Total Correcto"

            #if len(total_facturas)==0:
            #    col23.write(f"No hay errores en las facturas actuales y la suma total es de : {np.sum(totales_pred)}")
            #else:
            #    col23.write(f"Las facturas con valores de: {total_facturas} tienen totales diferentes.")
            #    col23.write(totales_pred)
            
            DF=pd.DataFrame({'Validación': validacion,'Facturas originales':total_facturas,'Totales no encontrados':totales_revisados})
            col23.write("Estos son los totales correctos o incorrectos, junto a las facturas originales y los totales identificados:")
            col23.dataframe(DF)
            #col23.download_button("Descargar como archivo de Excel", data=DF, file_name="Totales.xlsx")
        else:
            col23.warning("No se adjunto ninguna factura o archivo de Excel.")    
