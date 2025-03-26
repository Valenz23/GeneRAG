import streamlit as st
import tkinter as tk
import os
import shutil

from tkinter import filedialog

from classes.chatbot import Chatbot


### Muestra el número de docs que hay en la base de datos ###
def get_existing_docs():
    db = st.session_state["chatbot"].get_vector_store()
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    st.markdown("---")
    st.markdown(f"Número de documentos existentes en la base de datos: {len(existing_ids)}")


### Abre una ventana para seleccionar la carpeta desde la que cargar los documentos ###
# https://medium.com/@kjavaman12/how-to-create-a-folder-selector-in-streamlit-e44816c06afd
def select_folder():     
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        st.session_state["folder_path"] = folder_path   # path del directorio
        if folder_path:
            populate_db()

    except Exception as e:
        st.error(f"Se produjo un error: {e}")


### Carga de documentos desde directorio en la base de datos ###
@st.dialog("Actualizando base de datos")
def populate_db():   
    try:
        with st.spinner("Realizando el proceso, espere  ...", show_time=True):

            folder_path = st.session_state["folder_path"]
            mydata = st.session_state["chatbot"].get_docs_directory()
            if not os.path.exists(mydata):
                os.makedirs(mydata)

            for file in os.listdir(folder_path):                
                if file.endswith(".pdf"):
                    src_path = os.path.join(folder_path, file)
                    dst_path = os.path.join(mydata, file)    
                    if not file in os.listdir(mydata):      
                        shutil.copy2(src_path, dst_path)

            st.session_state["chatbot"].populate_db(mydata)

        st.session_state["folder_path"] = None  # reset path del directorio
        st.success("¡Proceso finalizado con éxito!")

    except Exception as e:
        st.error(f"Se produjo un error: {e}")


### Carga de documentos desde fichero/s a la base de datos ###
@st.dialog("Actualizando base de datos")
def upload_files():
    try:
        with st.spinner("Realizando el proceso, espere  ...", show_time=True):

            file_uploader = st.session_state["file_uploader"]
            mydata = st.session_state["chatbot"].get_docs_directory()
            if not os.path.exists(mydata):
                os.makedirs(mydata)

            for file in file_uploader:  
                file_path = os.path.join(mydata, file.name)  
                if not file in os.listdir(mydata):
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                st.session_state["chatbot"].load_document(file_path)

        st.success("¡Proceso finalizado con éxito!")

    except Exception as e:
        st.error(f"Se produjo un error: {e}")

##############################################################################################


def update_page():    
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = Chatbot()

    st.set_page_config(page_title="Actualizar ", page_icon="images/icon_blue.png")
    st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

    st.header("Actualizar la base de datos")

    get_existing_docs()    
    
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        st.text_input(
            label="Carga desde directorio",
            value = st.session_state.get("folder_path", None),            
            disabled=True
        )
    with col2:        
        st.button("Seleccionar y Cargar", on_click=select_folder)    

    st.file_uploader(
        "Carga desde fichero/s",
        key="file_uploader",
        type="pdf",
        accept_multiple_files=True,
        on_change=upload_files
    )


if __name__ == "__main__":
    update_page()