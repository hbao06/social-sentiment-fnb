"""
Chuẩn bị toàn bộ artifact cho Streamlit dashboard: dữ liệu mẫu, model, kết quả đánh giá.
Dùng lại đúng RANDOM_STATE=42 / SAMPLE_SIZE=60000 như notebook để số liệu khớp 100%.
"""
import os, re, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_curve, auc, precision_recall_curve, average_precision_score)

RANDOM_STATE = 42
SAMPLE_SIZE = 60000
RAW_PATH = '/mnt/user-data/uploads/training_1600000_processed_noemoticon_csv.zip'
OUT_DATA = os.path.join(os.path.dirname(__file__), 'data')
OUT_MODEL = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_MODEL, exist_ok=True)

print('1/7 - Đang tải dữ liệu gốc...')
df_raw = pd.read_csv(RAW_PATH, compression='zip', encoding='latin-1', header=None,
                      names=['target', 'id', 'date', 'flag', 'user', 'text'])
df_raw['date_parsed'] = pd.to_datetime(df_raw['date'].str.replace(' PDT', ''),
                                        format='%a %b %d %H:%M:%S %Y', errors='coerce')
df_raw['sentiment'] = df_raw['target'].map({0: 'Tiêu cực', 4: 'Tích cực'})

print('2/7 - Lấy mẫu & làm sạch...')
df_neg = df_raw[df_raw['target'] == 0].sample(n=SAMPLE_SIZE // 2, random_state=RANDOM_STATE)
df_pos = df_raw[df_raw['target'] == 4].sample(n=SAMPLE_SIZE // 2, random_state=RANDOM_STATE)
df = pd.concat([df_neg, df_pos], ignore_index=True)
df['text_len'] = df['text'].str.len()


def clean_text(t):
    t = t.lower()
    t = re.sub(r'http\S+|www\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


df['clean'] = df['text'].apply(clean_text)

print('3/7 - Topic modeling (LDA 5 chủ đề)...')
STOPWORDS = set('''a an the is are was were be been being to of and or in on at for with as it its this that
i you he she they we my your his her our their me him them us not no so do did does have has had just get
got go going will would can could should im ive dont didnt cant youre theyre isnt arent u ur ok yeah lol amp rt
but now out all like day too today back want what some time from really don why still feel need'''.split())

cv = CountVectorizer(max_features=3000, stop_words='english', min_df=5, max_df=0.5)
dtm = cv.fit_transform(df['clean'])
lda = LatentDirichletAllocation(n_components=5, random_state=RANDOM_STATE, max_iter=15, learning_method='online')
doc_topic = lda.fit_transform(dtm)
df['dominant_topic'] = doc_topic.argmax(axis=1)

feat_names_lda = cv.get_feature_names_out()
topic_info = {}
for t in range(5):
    top_idx = lda.components_[t].argsort()[-8:][::-1]
    words = [feat_names_lda[j] for j in top_idx]
    weights = [float(lda.components_[t][j]) for j in top_idx]
    sub = df[df['dominant_topic'] == t]
    pct_pos = float((sub['target'] == 4).mean() * 100)
    topic_info[str(t)] = {
        'label': ', '.join(words[:3]),
        'top_words': words,
        'weights': weights,
        'pct_positive': round(pct_pos, 1),
        'pct_negative': round(100 - pct_pos, 1),
        'n_mentions': int(len(sub)),
    }
with open(os.path.join(OUT_DATA, 'topic_info.json'), 'w', encoding='utf-8') as f:
    json.dump(topic_info, f, ensure_ascii=False, indent=2)

print('4/7 - Train/test split + TF-IDF + huấn luyện 3 mô hình...')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['target'])
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)
X_train = vectorizer.fit_transform(train_df['clean'])
X_test = vectorizer.transform(test_df['clean'])
y_train, y_test = train_df['target'], test_df['target']

nb = MultinomialNB().fit(X_train, y_train)
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE).fit(X_train, y_train)
svc = LinearSVC(random_state=RANDOM_STATE, max_iter=2000).fit(X_train, y_train)

pred_nb, pred_lr, pred_svc = nb.predict(X_test), lr.predict(X_test), svc.predict(X_test)
score_nb = nb.predict_proba(X_test)[:, 1]
score_lr = lr.predict_proba(X_test)[:, 1]
score_svc = svc.decision_function(X_test)


def metrics(y_true, y_pred):
    return {'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, pos_label=4),
            'Recall': recall_score(y_true, y_pred, pos_label=4),
            'F1': f1_score(y_true, y_pred, pos_label=4)}


model_comparison = {
    'Naive Bayes (baseline)': metrics(y_test, pred_nb),
    'Logistic Regression (nâng cao)': metrics(y_test, pred_lr),
    'Linear SVM (nâng cao)': metrics(y_test, pred_svc),
}

roc_data = {}
for name, score in [('Naive Bayes', score_nb), ('Logistic Regression', score_lr), ('Linear SVM', score_svc)]:
    fpr, tpr, _ = roc_curve(y_test, score, pos_label=4)
    prec, rec, _ = precision_recall_curve(y_test, score, pos_label=4)
    roc_data[name] = {
        'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc(fpr, tpr)),
        'precision': prec.tolist(), 'recall': rec.tolist(),
        'ap': float(average_precision_score(y_test, score, pos_label=4)),
    }

print('5/7 - Kiểm định McNemar & Cross-Validation...')
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

correct_lr = (pred_lr == y_test).values
correct_svc = (pred_svc == y_test).values
table = [[int(np.sum(correct_lr & correct_svc)), int(np.sum(correct_lr & ~correct_svc))],
         [int(np.sum(~correct_lr & correct_svc)), int(np.sum(~correct_lr & ~correct_svc))]]
mcnemar_res = mcnemar(table, exact=False, correction=True)

cv_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3)),
                         ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(cv_pipeline, df['clean'], df['target'], cv=skf, scoring=['accuracy', 'f1_macro'])

stats_summary = {
    'mcnemar_statistic': float(mcnemar_res.statistic), 'mcnemar_pvalue': float(mcnemar_res.pvalue),
    'cv_accuracy_scores': cv_results['test_accuracy'].tolist(),
    'cv_accuracy_mean': float(cv_results['test_accuracy'].mean()),
    'cv_accuracy_std': float(cv_results['test_accuracy'].std()),
    'cv_f1_scores': cv_results['test_f1_macro'].tolist(),
}

print('6/7 - Diễn giải mô hình (đóng góp tuyến tính từng từ, tương đương SHAP)...')
mean_train = np.asarray(X_train.mean(axis=0)).ravel()   # baseline cho công thức: contrib_i = coef_i * (x_i - mean_i)
feat_names_tfidf = vectorizer.get_feature_names_out()
coefs = lr.coef_[0]

# top feature theo |coef * (x_i - mean)| trung bình trên tập test (đã tính ở notebook) — lưu sẵn top 20 để vẽ nhanh
X_test_present = (X_test > 0).toarray()
contrib_matrix = X_test.toarray() * coefs - mean_train * coefs  # (x_i - mean_i) * coef_i cho từng doc
mean_abs = np.abs(contrib_matrix).mean(axis=0)
top_idx = np.argsort(mean_abs)[-20:]
directional = []
for i in top_idx:
    mask = X_test_present[:, i]
    cond_mean = contrib_matrix[mask, i].mean() if mask.sum() > 0 else 0.0
    directional.append({'word': feat_names_tfidf[i], 'value': float(cond_mean)})
directional = sorted(directional, key=lambda r: r['value'])
with open(os.path.join(OUT_DATA, 'shap_importance.json'), 'w', encoding='utf-8') as f:
    json.dump(directional, f, ensure_ascii=False, indent=2)

print('7/7 - Lưu artifacts...')
# dữ liệu cho Tab 1 & Tab 2 (đủ nhẹ để load nhanh)
mentions = df[['date_parsed', 'target', 'sentiment', 'text_len', 'dominant_topic']].copy()
mentions.columns = ['date', 'target', 'sentiment', 'text_len', 'dominant_topic']
mentions.to_parquet(os.path.join(OUT_DATA, 'mentions.parquet'), index=False)

# dữ liệu test cho Tab 3 (threshold slider, ROC/PR, confusion matrix, error table)
test_eval = test_df[['text']].copy()
test_eval['true_label'] = y_test.values
test_eval['pred_lr'] = pred_lr
test_eval['score_lr'] = score_lr
test_eval['score_nb'] = score_nb
test_eval['score_svc'] = score_svc
test_eval.to_parquet(os.path.join(OUT_DATA, 'test_eval.parquet'), index=False)

with open(os.path.join(OUT_DATA, 'model_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump({'comparison': model_comparison, 'roc_pr': roc_data, 'stats': stats_summary,
               'data_updated': str(df['date_parsed'].max()), 'n_mentions': int(len(df))}, f, indent=2)

joblib.dump(vectorizer, os.path.join(OUT_MODEL, 'vectorizer.pkl'))
joblib.dump(lr, os.path.join(OUT_MODEL, 'lr_model.pkl'))
np.save(os.path.join(OUT_MODEL, 'mean_train.npy'), mean_train)

print('XONG. Artifacts đã lưu tại', OUT_DATA, 'và', OUT_MODEL)
