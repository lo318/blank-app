import streamlit as st
import tempfile
import cv2
import os

st.set_page_config(page_title="GT7 Line Analyzer", layout="wide")
st.title("GT7 Line Analyzer - Protótipo Web com Análise Automática")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload de Vídeo da Corrida")
    video_file = st.file_uploader("Carrega um vídeo gravado na PS5 (qualquer formato)", type=None)

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        _, frame = cap.read()

        if frame is None:
            st.error("Erro ao ler o vídeo.")
        else:
            altura, largura, _ = frame.shape
            fps = cap.get(cv2.CAP_PROP_FPS)
            bbox = (largura // 2 - 50, altura // 2 - 50, 100, 100)

            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)

            caminho = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(tempfile.gettempdir(), "linha_detectada.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (largura, altura))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                sucesso, bbox = tracker.update(frame)
                if sucesso:
                    x, y, w, h = [int(v) for v in bbox]
                    centro = (x + w // 2, y + h // 2)
                    caminho.append(centro)
                    for i in range(1, len(caminho)):
                        cv2.line(frame, caminho[i - 1], caminho[i], (0, 255, 255), 2)

                out.write(frame)

            cap.release()
            out.release()

            st.success("Análise concluída!")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("Descarregar vídeo com linha traçada", f, "linha_detectada.mp4")

with col2:
    st.header("Notas de Análise")
    if video_file:
        st.markdown("- A trajetória foi estimada com base no seguimento automático do carro.")
        st.markdown("- Em futuras versões, será possível melhorar a precisão da análise com IA.")
    else:
        st.info("Carrega um vídeo para iniciar a análise.")
