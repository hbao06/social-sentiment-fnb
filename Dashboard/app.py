import os, json, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

import theme
from theme import COLORS, CATEGORICAL, CHART_CONFIG, icon, chip, raw_html, rgba

# ============================================================================
# CẤU HÌNH TRANG & TẢI DỮ LIỆU  (không đổi so với bản gốc)
# ============================================================================
st.set_page_config(page_title="Social Listening · F&B Sentiment", page_icon="📡", layout="wide",
                    initial_sidebar_state="expanded")

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, 'data')
MODELS = os.path.join(BASE, 'models')

SENT_COLOR = {"Tích cực": COLORS["positive"], "Tiêu cực": COLORS["negative"]}


@st.cache_data
def load_data():
    mentions = pd.read_parquet(os.path.join(DATA, 'mentions.parquet'))
    mentions['date'] = pd.to_datetime(mentions['date'])
    mentions['day'] = mentions['date'].dt.date
    test_eval = pd.read_parquet(os.path.join(DATA, 'test_eval.parquet'))
    with open(os.path.join(DATA, 'topic_info.json'), encoding='utf-8') as f:
        topic_info = json.load(f)
    with open(os.path.join(DATA, 'model_metrics.json'), encoding='utf-8') as f:
        model_metrics = json.load(f)
    with open(os.path.join(DATA, 'shap_importance.json'), encoding='utf-8') as f:
        shap_importance = json.load(f)
    return mentions, test_eval, topic_info, model_metrics, shap_importance


@st.cache_resource
def load_model():
    vectorizer = joblib.load(os.path.join(MODELS, 'vectorizer.pkl'))
    lr = joblib.load(os.path.join(MODELS, 'lr_model.pkl'))
    mean_train = np.load(os.path.join(MODELS, 'mean_train.npy'))
    return vectorizer, lr, mean_train


mentions, test_eval, topic_info, model_metrics, shap_importance = load_data()
vectorizer, lr_model, mean_train = load_model()

TOPIC_LABELS = {int(k): f"Chủ đề {int(k)+1} · {v['label']}" for k, v in topic_info.items()}

theme.inject_css()

# ============================================================================
# TINH CHINH SIDEBAR
# - An nut dong/mo sidebar mac dinh cua Streamlit.
# - Dua logo Social Listening len sat dau sidebar.
# ============================================================================
st.markdown("""
<style>
/* An nut mui ten dong/mo sidebar mac dinh cua Streamlit */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Giam khoang trong dau sidebar de logo nam o vi tri dau tien */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.75rem !important;
}

/* Logo thuong hieu nam gon va canh deu voi noi dung sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.sb-brand-title) {
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NAVIGATION STATE
# ============================================================================
PAGES = [
    {"key": "overview", "label": "Tổng quan", "icon": "dashboard",
     "title": "Tổng quan", "subtitle": None},
    {"key": "topics", "label": "Chủ đề & Từ khóa", "icon": "loyalty",
     "title": "Chủ đề & Từ khóa", "subtitle": None},
    {"key": "model", "label": "Hiệu năng mô hình", "icon": "analytics",
     "title": "Hiệu năng mô hình", "subtitle": None},
    {"key": "predict", "label": "Thử dự đoán trực tiếp", "icon": "psychology",
     "title": "Thử dự đoán trực tiếp", "subtitle": None},
]
PAGE_BY_KEY = {p["key"]: p for p in PAGES}

if "page" not in st.session_state:
    st.session_state.page = "overview"

# ============================================================================
# SIDEBAR
# ============================================================================
min_d, max_d = mentions['day'].min(), mentions['day'].max()

with st.sidebar:
    with st.container(key="sb_brand_row"):
        raw_html(f'<div style="display:flex; align-items:center; gap:8px;">'
                 f'{icon("satellite_alt", COLORS["accent"], 20)}'
                 f'<span class="sb-brand-title">Social Listening</span></div>')

    for p in PAGES:
        is_active = st.session_state.page == p["key"]
        if st.button(p["label"], key=f"nav_{p['key']}", icon=f":material/{p['icon']}:",
                     type="primary" if is_active else "secondary", use_container_width=True):
            st.session_state.page = p["key"]
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sb-section-label">BỘ LỌC</div>', unsafe_allow_html=True)

    date_col_from, date_col_to = st.columns(2)
    with date_col_from:
        date_from = st.date_input("Từ ngày", value=min_d, min_value=min_d, max_value=max_d,
                                   format="DD/MM/YYYY", key="f_date_from")
    with date_col_to:
        date_to = st.date_input("Đến ngày", value=max_d, min_value=min_d, max_value=max_d,
                                 format="DD/MM/YYYY", key="f_date_to")
    if date_from > date_to:
        st.warning("'Từ ngày' đang sau 'Đến ngày' — tạm dùng 'Từ ngày' cho cả 2 mốc.", icon="⚠️")
        date_range = (date_from, date_from)
    else:
        date_range = (date_from, date_to)

    topic_options = ["Tất cả chủ đề"] + [TOPIC_LABELS[k] for k in sorted(TOPIC_LABELS)]
    topic_choice = st.selectbox("Chủ đề (LDA)", options=topic_options, key="f_topic")
    sentiment_filter = st.multiselect("Cảm xúc", options=["Tích cực", "Tiêu cực"],
                                       default=["Tích cực", "Tiêu cực"], key="f_sentiment")

    if st.button("↺ Đặt lại bộ lọc", use_container_width=True, key="reset_filters"):
        for k in ("f_date_from", "f_date_to", "f_sentiment", "f_topic"):
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sb-section-label">DỮ LIỆU</div>', unsafe_allow_html=True)
    raw_html(f"""
    <div class="sb-dataset-card">
        <div class="sb-dataset-icon-wrap">{icon('database', COLORS['accent'], 15)}</div>
        <div class="sb-dataset-text">
            Cập nhật đến <b>{model_metrics['data_updated'][:10]}</b><br>
            Cỡ mẫu: <b>{model_metrics['n_mentions']:,}</b> mention
        </div>
    </div>
    """)

# ============================================================================
# LỌC DỮ LIỆU THEO SIDEBAR (áp dụng cho Tổng quan & Chủ đề)  — không đổi
# ============================================================================
mask = (mentions['day'] >= date_range[0]) & (mentions['day'] <= date_range[1])
mask &= mentions['sentiment'].isin(sentiment_filter)
if topic_choice != "Tất cả chủ đề":
    chosen_topic_id = [k for k, v in TOPIC_LABELS.items() if v == topic_choice][0]
    mask &= mentions['dominant_topic'] == chosen_topic_id
fdf = mentions.loc[mask]

with st.sidebar:
    st.download_button(
        "⬇ Tải dữ liệu đã lọc (CSV)",
        data=fdf.drop(columns=['day']).to_csv(index=False).encode('utf-8-sig'),
        file_name=f"mentions_loc_{len(fdf)}.csv", mime="text/csv", use_container_width=True,
        key="export_csv",
    )

active = PAGE_BY_KEY[st.session_state.page]
header_chips = [
    chip("filter_alt", f"{len(fdf):,} mention sau lọc"),
    chip("calendar_today", f"{date_range[0].strftime('%d/%m/%y')} – {date_range[1].strftime('%d/%m/%y')}"),
]
theme.page_header(active["title"], active["subtitle"], header_chips)

panel_start = theme.panel_start
panel_note = theme.panel_note
panel_end = theme.panel_end
style_chart = theme.style_chart

# ============================================================================
# TỔNG QUAN
# ============================================================================
if st.session_state.page == "overview":
    n_total = len(fdf)
    n_pos = int((fdf['sentiment'] == 'Tích cực').sum())
    n_neg = int((fdf['sentiment'] == 'Tiêu cực').sum())
    pct_neg = (n_neg / n_total * 100) if n_total else 0

    with st.container(key="kpi_row"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: theme.kpi_card("forum", f"{n_total:,}", "Tổng mention", "neutral")
        with c2: theme.kpi_card("thumb_up", f"{n_pos:,}", "Tích cực", "positive")
        with c3: theme.kpi_card("thumb_down", f"{n_neg:,}", "Tiêu cực", "negative")
        with c4: theme.kpi_card("trending_down", f"{pct_neg:.1f}%", "Tỷ lệ tiêu cực",
                                 "negative" if pct_neg > 50 else "positive")

    colA, colB = st.columns([1, 1.6])

    with colA:
        panel_start("Tỷ trọng cảm xúc", subtitle="Tích cực / Tiêu cực trong bộ lọc hiện tại")
        if n_total > 0:
            fig = go.Figure(data=[go.Pie(labels=list(SENT_COLOR.keys()),
                                          values=[n_pos, n_neg],
                                          hole=0.62,
                                          marker=dict(colors=[COLORS['positive'], COLORS['negative']],
                                                      line=dict(color=COLORS['surface'], width=2)),
                                          textinfo='percent', textfont=dict(size=13, color='#FFFFFF'))])
            style_chart(fig, height=150, showlegend=True, legend=dict(orientation='h', y=-0.12))
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            majority = "Tích cực" if n_pos >= n_neg else "Tiêu cực"
            panel_note(f"{majority} đang chiếm ưu thế với "
                       f"{max(n_pos, n_neg) / n_total * 100:.1f}% trong {n_total:,} mention đã lọc.")
        else:
            theme.empty_state("Không có dữ liệu khớp bộ lọc hiện tại.")
        panel_end()

    with colB:
        panel_start("Diễn biến % Tiêu cực theo ngày", subtitle="Kéo bộ lọc khoảng ngày ở sidebar để phóng to giai đoạn quan tâm")
        daily = fdf.groupby(['day', 'sentiment']).size().unstack(fill_value=0)
        if 'Tiêu cực' in daily.columns and 'Tích cực' in daily.columns and len(daily) > 0:
            daily['pct_neg'] = daily.get('Tiêu cực', 0) / (daily.get('Tiêu cực', 0) + daily.get('Tích cực', 0)).replace(0, np.nan) * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(daily.index), y=daily['pct_neg'], mode='lines',
                                      line=dict(color=COLORS['negative'], width=2.5),
                                      fill='tozeroy', fillcolor=rgba(COLORS['negative'], 0.08),
                                      name='% Tiêu cực', hovertemplate='%{x|%d/%m/%Y}: %{y:.1f}%<extra></extra>'))
            fig.add_hline(y=50, line_dash='dash', line_color=COLORS['text_faint'], line_width=1)
            peak_day = daily['pct_neg'].idxmax()
            peak_val = daily['pct_neg'].max()
            fig.add_annotation(x=peak_day, y=peak_val, text=f"Đỉnh {peak_val:.0f}%",
                                showarrow=True, arrowhead=2, arrowcolor=COLORS['negative'],
                                font=dict(color=COLORS['negative'], size=11), ay=-28)
            style_chart(fig, height=150, xaxis_title=None, yaxis_title='% Tiêu cực', showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            panel_note(f"Tỷ lệ tiêu cực cao nhất vào {pd.Timestamp(peak_day).strftime('%d/%m/%Y')} "
                       f"({peak_val:.1f}%) — đáng để đối chiếu với sự kiện/thông báo phát sinh ngày đó.")
        else:
            theme.empty_state("Cần cả 2 nhãn cảm xúc trong bộ lọc để vẽ xu hướng.")
        panel_end()

    colC, colD = st.columns([1, 1])

    with colC:
        panel_start("Phân phối độ dài văn bản", subtitle="Theo từng nhóm cảm xúc")
        if n_total > 0:
            fig = go.Figure()
            for s, color in SENT_COLOR.items():
                vals = fdf.loc[fdf['sentiment'] == s, 'text_len']
                if len(vals):
                    fig.add_trace(go.Histogram(x=vals, name=s, marker_color=color, opacity=0.65, nbinsx=30))
            fig.update_layout(barmode='overlay')
            style_chart(fig, height=150, xaxis_title='Số ký tự', yaxis_title='Số mention',
                        showlegend=True, legend=dict(orientation='h', y=-0.22))
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            len_pos = fdf.loc[fdf['sentiment'] == 'Tích cực', 'text_len'].mean()
            len_neg = fdf.loc[fdf['sentiment'] == 'Tiêu cực', 'text_len'].mean()
            if pd.notna(len_pos) and pd.notna(len_neg):
                if abs(len_neg - len_pos) < 2:
                    panel_note(f"Độ dài mention Tích cực và Tiêu cực gần như tương đương "
                               f"({len_pos:.1f} vs {len_neg:.1f} ký tự).")
                else:
                    longer = "Tiêu cực" if len_neg > len_pos else "Tích cực"
                    panel_note(f"Mention {longer} trung bình dài hơn "
                               f"({len_neg:.1f} vs {len_pos:.1f} ký tự).")
        else:
            theme.empty_state("Không có dữ liệu khớp bộ lọc hiện tại.")
        panel_end()

    with colD:
        panel_start("Mention theo khung giờ trong ngày", subtitle="Giờ nào cộng đồng nhắc đến thương hiệu nhiều nhất")
        if n_total > 0:
            hourly = fdf.groupby(fdf['date'].dt.hour).size().reindex(range(24), fill_value=0)
            fig = go.Figure(go.Bar(x=list(hourly.index), y=hourly.values, marker_color=COLORS['accent']))
            style_chart(fig, height=150, xaxis_title='Giờ trong ngày', yaxis_title='Số mention', showlegend=False)
            fig.update_xaxes(dtick=2)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            peak_hour = int(hourly.idxmax())
            panel_note(f"Khung giờ {peak_hour}:00–{(peak_hour+1)%24}:00 có nhiều mention nhất "
                       f"({int(hourly.max()):,}) — cân nhắc lên lịch phản hồi CSKH quanh khung giờ này.")
        else:
            theme.empty_state("Không có dữ liệu khớp bộ lọc hiện tại.")
        panel_end()

# ============================================================================
# CHỦ ĐỀ & TỪ KHÓA
# ============================================================================
elif st.session_state.page == "topics":
    topic_ids_all = sorted(topic_info, key=lambda k: int(k))
    most_negative_topic = max(topic_ids_all, key=lambda k: topic_info[k]['pct_negative'])
    largest_topic = max(topic_ids_all, key=lambda k: topic_info[k]['n_mentions'])

    colA, colB = st.columns([1.3, 1])

    with colA:
        panel_start("Từ khóa nổi bật theo chủ đề")
        topic_pick = st.selectbox("Chọn chủ đề để xem từ khóa:",
                                   options=sorted(TOPIC_LABELS, key=lambda k: k),
                                   format_func=lambda k: TOPIC_LABELS[k], key="topic_pick_tab2")
        info = topic_info[str(topic_pick)]
        fig = go.Figure(go.Bar(x=info['weights'][::-1], y=info['top_words'][::-1], orientation='h',
                                marker_color=COLORS['accent']))
        style_chart(fig, height=220, xaxis_title='Trọng số LDA')
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        panel_note(f"Chủ đề này chiếm {info['n_mentions']:,} mention trong mẫu, "
                   f"{info['pct_negative']:.1f}% tiêu cực / {info['pct_positive']:.1f}% tích cực.")
        panel_end()

    with colB:
        panel_start("Tỷ lệ cảm xúc theo từng chủ đề")
        topic_ids = sorted(topic_info, key=lambda k: int(k))
        labels = [f"Chủ đề {int(k)+1}" for k in topic_ids]
        pct_pos_list = [topic_info[k]['pct_positive'] for k in topic_ids]
        pct_neg_list = [topic_info[k]['pct_negative'] for k in topic_ids]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=labels, x=pct_pos_list, name='Tích cực', orientation='h',
                              marker_color=COLORS['positive']))
        fig.add_trace(go.Bar(y=labels, x=pct_neg_list, name='Tiêu cực', orientation='h',
                              marker_color=COLORS['negative']))
        style_chart(fig, height=220, barmode='stack', legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        panel_note(f"Chủ đề {int(most_negative_topic)+1} ({topic_info[most_negative_topic]['label']}) "
                   f"nghiêng tiêu cực nhất ({topic_info[most_negative_topic]['pct_negative']:.1f}%) "
                   f"— nên ưu tiên theo dõi.")
        panel_end()

    st.write("")
    panel_start("Quy mô các chủ đề trong toàn bộ dữ liệu", subtitle="Diện tích tỷ lệ với số mention — màu thể hiện % tiêu cực")
    fig = go.Figure(go.Treemap(
        labels=[f"Chủ đề {int(k)+1}<br>{topic_info[k]['label']}" for k in topic_ids_all],
        parents=["" for _ in topic_ids_all],
        values=[topic_info[k]['n_mentions'] for k in topic_ids_all],
        marker=dict(colors=[topic_info[k]['pct_negative'] for k in topic_ids_all],
                    colorscale=[[0, COLORS['positive']], [1, COLORS['negative']]],
                    cmin=30, cmax=70, showscale=True,
                    colorbar=dict(title='% Tiêu cực', thickness=14)),
        texttemplate="<b>%{label}</b><br>%{value:,} mention",
        textfont=dict(size=12, color='white'),
    ))
    style_chart(fig, height=200, margin=dict(t=8, b=8, l=8, r=8))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    panel_note(f"Chủ đề {int(largest_topic)+1} ({topic_info[largest_topic]['label']}) lớn nhất với "
               f"{topic_info[largest_topic]['n_mentions']:,} mention "
               f"({topic_info[largest_topic]['n_mentions']/sum(topic_info[k]['n_mentions'] for k in topic_ids_all)*100:.1f}% tổng dữ liệu).")
    panel_end()

# ============================================================================
# HIỆU NĂNG MÔ HÌNH
# ============================================================================
elif st.session_state.page == "model":
    best_comp = model_metrics['comparison']['Logistic Regression (nâng cao)']
    best_roc = model_metrics['roc_pr']['Logistic Regression']
    with st.container(key="kpi_row"):
        k1, k2, k3, k4 = st.columns(4)
        with k1: theme.kpi_card("verified", f"{best_comp['Accuracy']*100:.1f}%", "Accuracy (LR)", "accent")
        with k2: theme.kpi_card("balance", f"{best_comp['F1']*100:.1f}%", "F1-score (LR)", "accent")
        with k3: theme.kpi_card("show_chart", f"{best_roc['auc']:.3f}", "AUC-ROC (LR)", "positive")
        with k4: theme.kpi_card("database", f"{len(test_eval):,}", "Mention tập test", "neutral")

    colA, colB = st.columns([1, 1])
    with colA:
        panel_start("So sánh 3 mô hình")
        comp_df = pd.DataFrame(model_metrics['comparison']).T
        fig = go.Figure()
        model_colors = {
            'Naive Bayes (baseline)': CATEGORICAL[1],
            'Logistic Regression (nâng cao)': COLORS['accent'],
            'Linear SVM (nâng cao)': CATEGORICAL[2],
        }
        for model_name in comp_df.index:
            fig.add_trace(go.Bar(x=comp_df.columns, y=comp_df.loc[model_name],
                                  name=model_name, marker_color=model_colors[model_name]))
        style_chart(fig, height=210, barmode='group', yaxis_range=[0, 1],
                    legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        panel_note(
            f"McNemar (LR vs SVM): p = {model_metrics['stats']['mcnemar_pvalue']:.2e} "
            f"→ khác biệt có ý nghĩa thống kê. 5-Fold CV Accuracy: {model_metrics['stats']['cv_accuracy_mean']*100:.2f}% "
            f"± {model_metrics['stats']['cv_accuracy_std']*100:.2f}%.")
        panel_end()

    with colB:
        panel_start("Ngưỡng phân loại tương tác (Logistic Regression)")
        threshold = st.slider("Ngưỡng xác suất để gán nhãn 'Tích cực'", 0.05, 0.95, 0.50, 0.01)
        y_true = test_eval['true_label'].values
        y_pred_thresh = np.where(test_eval['score_lr'].values >= threshold, 4, 0)
        tp = int(np.sum((y_pred_thresh == 4) & (y_true == 4)))
        fp = int(np.sum((y_pred_thresh == 4) & (y_true == 0)))
        fn = int(np.sum((y_pred_thresh == 0) & (y_true == 4)))
        tn = int(np.sum((y_pred_thresh == 0) & (y_true == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        cm = np.array([[tn, fp], [fn, tp]])
        fig = go.Figure(data=go.Heatmap(z=cm, x=['Tiêu cực', 'Tích cực'], y=['Tiêu cực', 'Tích cực'],
                                         colorscale=[[0, COLORS['surface_alt']], [1, COLORS['accent']]],
                                         text=cm, texttemplate="%{text:,}",
                                         textfont=dict(size=13), showscale=False))
        style_chart(fig, height=200, xaxis_title='Dự đoán', yaxis_title='Thực tế')
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        panel_note(f"Ở ngưỡng {threshold:.2f}: Precision = {prec*100:.1f}% · Recall = {rec*100:.1f}%")
        panel_end()

    colC, colD = st.columns([1, 1])
    with colC:
        panel_start("Đường cong ROC — 3 mô hình")
        fig = go.Figure()
        for name, color in [('Naive Bayes', CATEGORICAL[1]), ('Logistic Regression', COLORS['accent']),
                             ('Linear SVM', CATEGORICAL[2])]:
            r = model_metrics['roc_pr'][name]
            fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines', name=f"{name} (AUC={r['auc']:.3f})",
                                      line=dict(color=color, width=2.2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  line=dict(dash='dash', color=COLORS['text_faint'], width=1),
                                  showlegend=False))
        style_chart(fig, height=210, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                    legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        best_name = max(['Naive Bayes', 'Logistic Regression', 'Linear SVM'],
                         key=lambda n: model_metrics['roc_pr'][n]['auc'])
        panel_note(f"{best_name} có AUC cao nhất ({model_metrics['roc_pr'][best_name]['auc']:.3f}) "
                   f"— khả năng phân tách 2 lớp tốt nhất trong 3 mô hình.")
        panel_end()

    with colD:
        panel_start("Đường cong Precision-Recall — 3 mô hình")
        fig = go.Figure()
        for name, color in [('Naive Bayes', CATEGORICAL[1]), ('Logistic Regression', COLORS['accent']),
                             ('Linear SVM', CATEGORICAL[2])]:
            r = model_metrics['roc_pr'][name]
            fig.add_trace(go.Scatter(x=r['recall'], y=r['precision'], mode='lines',
                                      name=f"{name} (AP={r['ap']:.3f})", line=dict(color=color, width=2.2)))
        style_chart(fig, height=210, xaxis_title='Recall', yaxis_title='Precision',
                    legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        panel_note("PR curve phù hợp hơn ROC khi quan tâm cân bằng giữa bắt đúng (recall) và "
                   "tránh báo sai (precision) — đặc biệt hữu ích nếu 2 lớp cảm xúc không đồng đều.")
        panel_end()

    panel_start("Diễn giải mô hình — từ khóa ảnh hưởng mạnh nhất", subtitle="Logistic Regression · hệ số hồi quy trung bình")
    words = [d['word'] for d in shap_importance]
    vals = [d['value'] for d in shap_importance]
    colors_bar = [COLORS['negative'] if v < 0 else COLORS['positive'] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=words, orientation='h', marker_color=colors_bar))
    style_chart(fig, height=210, xaxis_title='Đóng góp trung bình (âm=Tiêu cực, dương=Tích cực)')
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    panel_note("Tương đương phương pháp SHAP cho mô hình tuyến tính — xem chi tiết cách tính trong notebook mục 4.")
    panel_end()

    panel_start("Các trường hợp dự đoán sai — độ tự tin cao nhất")
    wrong = test_eval[test_eval['pred_lr'] != test_eval['true_label']].copy()
    wrong['confidence'] = np.where(wrong['pred_lr'] == 4, wrong['score_lr'], 1 - wrong['score_lr'])
    wrong = wrong.sort_values('confidence', ascending=False).head(15)
    wrong_display = wrong[['text']].copy()
    wrong_display['Thực tế'] = np.where(wrong['true_label'] == 4, 'Tích cực', 'Tiêu cực')
    wrong_display['Dự đoán'] = np.where(wrong['pred_lr'] == 4, 'Tích cực', 'Tiêu cực')
    wrong_display['Độ tự tin'] = (wrong['confidence'] * 100).round(1).astype(str) + '%'
    wrong_display.columns = ['Nội dung mention', 'Thực tế', 'Dự đoán', 'Độ tự tin']
    st.dataframe(
    wrong_display,
    use_container_width=True,
    hide_index=True,
    height=250
)
    panel_note("Bảng có thể sort theo cột (bấm tiêu đề cột) — dùng để rà soát định tính các ca mô hình dễ nhầm.")
    panel_end()

# ============================================================================
# THỬ DỰ ĐOÁN TRỰC TIẾP
# ============================================================================
elif st.session_state.page == "predict":
    st.session_state.setdefault("predict_history", [])

    panel_start("Nhập một mention để mô hình dự đoán cảm xúc",
                subtitle="Mô hình được huấn luyện trên dữ liệu tiếng Anh")

    examples = [
        "I love this new drink, best thing I've tried all year!",
        "Waited 40 minutes and they still got my order wrong, never again.",
        "It's okay, nothing special but not bad either.",
    ]
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(f"Ví dụ {i + 1}", key=f"ex_{i}", use_container_width=True, help=ex):
                st.session_state["predict_input"] = ex
                st.rerun()

    user_text = st.text_area("Nội dung mention:",
                              placeholder="VD: I love this new drink, best thing I've tried all year!",
                              height=90, label_visibility="collapsed", key="predict_input")
    run = st.button("Phân tích cảm xúc", type="primary", icon=":material/insights:")
    panel_end()

    if run and user_text.strip():
        def clean_text(t):
            t = t.lower()
            t = re.sub(r'http\S+|www\S+', '', t)
            t = re.sub(r'@\w+', '', t)
            t = re.sub(r'[^a-z\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        cleaned = clean_text(user_text)
        x_vec = vectorizer.transform([cleaned])
        proba = lr_model.predict_proba(x_vec)[0]
        pred_label = 'Tích cực' if proba[1] >= 0.5 else 'Tiêu cực'
        confidence = max(proba)

        st.session_state.predict_history.insert(
            0, {"text": user_text.strip(), "label": pred_label, "confidence": float(confidence)})
        st.session_state.predict_history = st.session_state.predict_history[:5]

        colA, colB = st.columns([1, 1.4])
        with colA:
            panel_start("Kết quả dự đoán")
            color = COLORS['positive'] if pred_label == 'Tích cực' else COLORS['negative']
            soft = COLORS['positive_soft'] if pred_label == 'Tích cực' else COLORS['negative_soft']
            ic = 'sentiment_satisfied' if pred_label == 'Tích cực' else 'sentiment_dissatisfied'
            raw_html(f"""
            <div class="predict-result">
                <div style="font-size:0.78rem; color:{COLORS['text_muted']}; font-weight:600;">CẢM XÚC DỰ ĐOÁN</div>
                <div class="predict-badge" style="background:{soft}; color:{color};">
                    {icon(ic, color, 20)} {pred_label}
                </div>
                <div class="predict-confidence">Độ tin cậy: <b style="color:{COLORS['text']};">{confidence*100:.1f}%</b></div>
            </div>
            """)
            fig = go.Figure(go.Bar(x=[proba[1]*100], y=[''], orientation='h', marker_color=COLORS['positive'],
                                    name='Tích cực', width=0.5))
            fig.add_trace(go.Bar(x=[proba[0]*100], y=[''], orientation='h', marker_color=COLORS['negative'],
                                  name='Tiêu cực', width=0.5))
            style_chart(fig, height=90, barmode='stack', showlegend=True,
                        xaxis=dict(range=[0, 100], title='%'), legend=dict(orientation='h', y=-0.6))
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            panel_end()

        with colB:
            panel_start("Tín hiệu chính", subtitle="Từ nào trong câu ảnh hưởng đến dự đoán")
            feat_names = vectorizer.get_feature_names_out()
            x_arr = x_vec.toarray().ravel()
            coefs = lr_model.coef_[0]
            present_idx = np.nonzero(x_arr)[0]
            contribs = [(feat_names[i], float(coefs[i] * (x_arr[i] - mean_train[i]))) for i in present_idx]
            contribs = sorted(contribs, key=lambda r: r[1])
            if contribs:
                words_c = [c[0] for c in contribs]
                vals_c = [c[1] for c in contribs]
                colors_c = [COLORS['negative'] if v < 0 else COLORS['positive'] for v in vals_c]
                fig = go.Figure(go.Bar(x=vals_c, y=words_c, orientation='h', marker_color=colors_c))
                style_chart(fig, height=max(140, min(220, 28 * len(contribs))),
                            xaxis_title='Đóng góp vào điểm dự đoán')
                st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            else:
                theme.empty_state(
                    "Không có từ/cụm từ nào trong câu khớp với từ điển 10.000 đặc trưng đã học — "
                    "mô hình dự đoán chủ yếu dựa vào baseline (intercept).", icon_name="text_fields")
            panel_note("Công thức: đóng góp = hệ số hồi quy × (giá trị TF-IDF của từ trong câu − giá trị trung bình trên tập huấn luyện) — chính là cách SHAP tính cho mô hình tuyến tính.")
            panel_end()
    elif run:
        st.warning("Vui lòng nhập nội dung trước khi dự đoán.")

    if st.session_state.predict_history:
        st.write("")
        panel_start("Lịch sử dự đoán trong phiên này", subtitle=f"{len(st.session_state.predict_history)} lượt gần nhất")
        hist_df = pd.DataFrame(st.session_state.predict_history)
        hist_df["confidence"] = (hist_df["confidence"] * 100).round(1).astype(str) + "%"
        hist_df.columns = ["Nội dung mention", "Dự đoán", "Độ tin cậy"]
        st.dataframe(
            hist_df,
            use_container_width=True,
            hide_index=True
        )
        panel_end()