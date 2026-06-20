import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import pandas as pd
import re
import plotly.express as px

st.set_page_config(
    page_title="VisionGuard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body { background-color: #0d0d0d; }
    .block-container { padding: 2rem 3rem; }
    h1 { color: #FF4B4B; letter-spacing: -1px; font-size: 2.4rem; }
    .metric-box {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-box h2 { font-size: 2rem; margin: 0; }
    .metric-box p  { color: #888; margin: 0; font-size: 0.85rem; }
    .violation-tag {
        background: #3d0000;
        border-left: 3px solid #FF4B4B;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .plate-tag {
        background: #001a33;
        border-left: 3px solid #4B9EFF;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 1rem;
    }
    .perf-box {
        background: #0a1a0a;
        border: 1px solid #1a3a1a;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .perf-box h3 { color: #00CC66; margin: 0; font-size: 1.6rem; }
    .perf-box p  { color: #888; margin: 0; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    helmet = YOLO(r'C:\Users\manis\OneDrive\Desktop\Visionguard\models\runs\detect\visionguard_helmet-4\weights\best.pt')
    plate  = YOLO(r'C:\Users\manis\OneDrive\Desktop\Visionguard\models\runs\detect\visionguard_plate\weights\best.pt')
    tl     = YOLO(r'C:\Users\manis\OneDrive\Desktop\Visionguard\models\runs\detect\visionguard_traffic_light\weights\best.pt')
    ocr    = easyocr.Reader(['en'], gpu=False)
    return helmet, plate, tl, ocr

helmet_model, plate_model, tl_model, reader = load_models()

def detect_red_light_violation(image):
    tl_results = tl_model(image)[0]
    violations = []
    is_red = False

    for box in tl_results.boxes:
        if int(box.cls) == 1:  # Red
            is_red = True
            break

    if is_red:
        plate_results = plate_model(image)[0]
        for box in plate_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image[y1:y2, x1:x2]
            ocr_result = reader.readtext(plate_crop)
            plate_text = " ".join([r[1] for r in ocr_result]) if ocr_result else "Unreadable"
            violations.append({
                "type": "Red Light Violation",
                "plate": plate_text,
                "confidence": f"{float(box.conf):.2f}"
            })

    return violations, is_red

LOG_PATH = 'logs/violations.csv'
os.makedirs('logs', exist_ok=True)

def log_violation(plate, v_type, confidence):
    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            plate, v_type, f"{confidence:.1%}"
        ])

def read_plate(img, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    result = reader.readtext(gray)
    if result:
        text = re.sub(r'[^A-Z0-9]', '', result[0][1].upper())
        return text if len(text) > 3 else None
    return None

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def check_triple_riding(boxes, names, img_width):
    riders = []
    for box in boxes:
        if names[int(box.cls)] in ['With Helmet', 'Without Helmet']:
            x_center = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            riders.append(x_center)

    if len(riders) < 3:
        return False, len(riders)

    riders.sort()
    for i in range(len(riders) - 2):
        if riders[i+2] - riders[i] < img_width * 0.30:
            return True, len(riders)

    return False, len(riders)

st.markdown("# 🚦 VisionGuard")
st.markdown("##### AI-Powered Traffic Violation Detection — Gridlock Hackathon 2.0")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics", "🎯 Model Performance"])

with tab1:
    uploaded = st.file_uploader("Upload a traffic image", type=['jpg','jpeg','png'])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = preprocess(img)

        with st.spinner("Analyzing image..."):
            helmet_results = helmet_model(img)
            plate_results  = plate_model(img)

            plates = [read_plate(img, box) for box in plate_results[0].boxes]
            plates = [p for p in plates if p]
            plate_str = plates[0] if plates else "Unknown"

            violations = []
            for box in helmet_results[0].boxes:
                label = helmet_results[0].names[int(box.cls)]
                conf  = float(box.conf)
                if label == 'Without Helmet':
                    violations.append((label, conf))
                    log_violation(plate_str, 'No Helmet', conf)

            is_triple, rider_count = check_triple_riding(
                helmet_results[0].boxes,
                helmet_results[0].names,
                img.shape[1]
            )
            if is_triple:
                log_violation(plate_str, 'Triple Riding', 0.95)

            # Red light detection
            tl_violations, is_red = detect_red_light_violation(img)
            if is_red:
                for v in tl_violations:
                    log_violation(v['plate'], 'Red Light Violation', float(v['confidence']))

        col1, col2 = st.columns([2, 1])

        with col1:
            annotated = helmet_results[0].plot()
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width=800)

        with col2:
            total_violations = len(violations) + (1 if is_triple else 0) + (1 if is_red else 0)
            st.markdown(f"""
            <div class="metric-box">
                <h2 style="color:{'#FF4B4B' if total_violations else '#00CC66'}">
                    {total_violations}
                </h2>
                <p>VIOLATION(S) DETECTED</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if plates:
                for p in plates:
                    st.markdown(f'<div class="plate-tag">🪪 {p}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="plate-tag">🪪 Plate not detected</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if violations:
                for label, conf in violations:
                    st.markdown(
                        f'<div class="violation-tag">⚠️ {label} — {conf:.0%} confidence</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("No helmet violations")

            if is_triple:
                st.markdown(
                    f'<div class="violation-tag">⚠️ Triple Riding — {rider_count} riders detected</div>',
                    unsafe_allow_html=True
                )

            if is_red:
                plate_info = tl_violations[0]['plate'] if tl_violations else "Unknown"
                st.markdown(
                    f'<div class="violation-tag">🔴 Red Light Violation — Plate: {plate_info}</div>',
                    unsafe_allow_html=True
                )

with tab2:
    if st.button("Clear Logs"):
        open(LOG_PATH, 'w').close()
    st.success("Logs cleared.")
    try:
        df = pd.read_csv(LOG_PATH, names=['Time','Plate','Violation','Confidence'])
        df['Time'] = pd.to_datetime(df['Time'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Violations", len(df))
        c2.metric("Unique Plates",    df['Plate'].nunique())
        c3.metric("Most Common",      df['Violation'].mode()[0] if len(df) else "—")
        c4.metric("Today",            len(df[df['Time'].dt.date == datetime.now().date()]))

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Violations by Type**")
            fig1 = px.bar(
                df['Violation'].value_counts().reset_index(),
                x='Violation', y='count',
                color='Violation',
                color_discrete_sequence=['#FF4B4B','#FF8C00','#FFD700'],
                template='plotly_dark'
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**Violations Over Time**")
            fig2 = px.histogram(
                df, x='Time',
                color='Violation',
                template='plotly_dark',
                color_discrete_sequence=['#FF4B4B','#FF8C00']
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Full Log**")
        st.dataframe(
            df.sort_values('Time', ascending=False),
            use_container_width=True
        )

    except Exception:
        st.info("No violations logged yet. Upload an image in the Detection tab.")

with tab3:
    st.markdown("### 🪖 Helmet Detection Model — YOLOv8n")
    st.caption("Fine-tuned on Indian traffic dataset | 30 epochs | RTX 2050")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="perf-box"><h3>90.2%</h3><p>mAP50</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="perf-box"><h3>90.0%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="perf-box"><h3>83.1%</h3><p>Recall</p></div>', unsafe_allow_html=True)
    c4.markdown('<div class="perf-box"><h3>53.3%</h3><p>mAP50-95</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Per-Class Breakdown**")
    perf_df = pd.DataFrame({
        'Class':     ['With Helmet', 'Without Helmet'],
        'Precision': [0.933, 0.867],
        'Recall':    [0.875, 0.788],
        'mAP50':     [0.940, 0.865]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    fig3 = px.bar(
        perf_df.melt(id_vars='Class', var_name='Metric', value_name='Score'),
        x='Metric', y='Score', color='Class', barmode='group',
        template='plotly_dark',
        color_discrete_sequence=['#00CC66','#FF4B4B']
    )
    fig3.update_yaxes(range=[0,1])
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    st.markdown("### 🪪 License Plate Detection Model — YOLOv8n")
    st.caption("Trained on Indian license plate dataset | 30 epochs | RTX 2050")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="perf-box"><h3>99.5%</h3><p>mAP50</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="perf-box"><h3>98.7%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="perf-box"><h3>99.4%</h3><p>Recall</p></div>', unsafe_allow_html=True)
    c4.markdown('<div class="perf-box"><h3>84.2%</h3><p>mAP50-95</p></div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🚦 Traffic Light Detection Model — YOLOv8n")
    st.caption("Trained on traffic light dataset (Red/Green/Yellow) | 30 epochs | RTX 2050")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="perf-box"><h3>96.6%</h3><p>mAP50</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="perf-box"><h3>94.9%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="perf-box"><h3>90.3%</h3><p>Recall</p></div>', unsafe_allow_html=True)
    c4.markdown('<div class="perf-box"><h3>71.8%</h3><p>mAP50-95</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Per-Class Breakdown**")
    tl_df = pd.DataFrame({
        'Class':     ['Green', 'Red', 'Yellow'],
        'Precision': [0.949, 0.952, 0.946],
        'Recall':    [0.901, 0.908, 0.900],
        'mAP50':     [0.968, 0.971, 0.958]
    })
    st.dataframe(tl_df, use_container_width=True, hide_index=True)
    fig4 = px.bar(
        tl_df.melt(id_vars='Class', var_name='Metric', value_name='Score'),
        x='Metric', y='Score', color='Class', barmode='group',
        template='plotly_dark',
        color_discrete_sequence=['#00CC66','#FF4B4B','#FFD700']
    )
    fig4.update_yaxes(range=[0,1])
    st.plotly_chart(fig4, use_container_width=True)