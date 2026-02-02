import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

# ==============================
# CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="GaitScan 3D Pro", layout="wide")
st.title("🏃 GaitScan 3D Pro (MoveNet Cloud)")
st.subheader("Analyse cinématique par stéréoscopie 3D simplifiée")

# ==============================
# MOVE NET
# ==============================
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def detect_pose(frame):
    """Détecte 17 points clés (y, x, score) avec MoveNet"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    keypoints = outputs['output_0'].numpy()  # [1,1,17,3]
    return keypoints[0,0,:,:]  # [17,3]

# Map des articulations que l'on veut suivre
JOINTS = {
    "Cheville G": 15,
    "Genou G": 13,
    "Hanche G": 11,
    "Cheville D": 16,
    "Genou D": 14,
    "Hanche D": 12
}

# ==============================
# TRAITEMENT VIDÉO
# ==============================
def process_video_pair(video_left, video_right, frame_skip=2):
    capL = cv2.VideoCapture(video_left)
    capR = cv2.VideoCapture(video_right)
    results_3d = {name: [] for name in JOINTS.keys()}
    
    frame_idx = 0
    while capL.isOpened() and capR.isOpened():
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break
        if frame_idx % frame_skip == 0:
            kpL = detect_pose(frameL)
            kpR = detect_pose(frameR)
            for name, idx in JOINTS.items():
                yL, xL, sL = kpL[idx]
                yR, xR, sR = kpR[idx]
                X = (xL + xR) / 2
                Y = (yL + yR) / 2
                Z = abs(xL - xR)
                results_3d[name].append([X, Y, Z])
        frame_idx += 1

    capL.release()
    capR.release()
    return results_3d

# ==============================
# EXPORT PDF
# ==============================
def export_gait_pdf(patient_info, joint_images):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_marche_3d.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>BILAN D'ANALYSE DE LA MARCHE 3D</b>", styles['Title']),
        Paragraph(f"Patient : {patient_info['nom']} {patient_info['prenom']}", styles['Normal']),
        Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']),
        Spacer(1, 1*cm)
    ]
    for joint_name, img_path in joint_images.items():
        story.append(Paragraph(f"<b>Articulation : {joint_name}</b>", styles['Heading2']))
        story.append(PDFImage(img_path, width=15*cm, height=8*cm))
        story.append(Spacer(1, 0.5*cm))
    doc.build(story)
    return path

# ==============================
# INTERFACE STREAMLIT
# ==============================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.subheader("📹 Vidéos")
    file_left = st.file_uploader("Angle Gauche", type=["mp4","avi","mov"])
    file_right = st.file_uploader("Angle Droit", type=["mp4","avi","mov"])

if file_left and file_right:
    if st.button("⚙️ LANCER L'ANALYSE"):
        with st.spinner("Analyse en cours..."):
            t_file_l = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_file_l.write(file_left.read())
            t_file_r = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_file_r.write(file_right.read())
            
            data_3d = process_video_pair(t_file_l.name, t_file_r.name, frame_skip=2)
            
            joint_imgs = {}
            for joint_name, coords in data_3d.items():
                if len(coords) > 5:
                    coords = np.array(coords)
                    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
                    ax1.plot(coords[:,0], -coords[:,1], color='red', lw=2)
                    ax1.set_title("Plan Frontal (X-Y)")
                    ax1.axis('off')
                    ax2.plot(coords[:,2], -coords[:,1], color='blue', lw=2)
                    ax2.set_title("Plan Sagittal (Z-Y)")
                    ax2.axis('off')
                    st.pyplot(fig)
                    img_path = os.path.join(tempfile.gettempdir(), f"{joint_name}.png")
                    fig.savefig(img_path, bbox_inches='tight')
                    plt.close(fig)
                    joint_imgs[joint_name] = img_path
            
            pdf_path = export_gait_pdf({"nom": nom, "prenom": prenom}, joint_imgs)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Télécharger PDF", f, f"Gait_3D_{nom}.pdf")
            
            # Nettoyage fichiers temporaires
            os.unlink(t_file_l.name)
            os.unlink(t_file_r.name)
