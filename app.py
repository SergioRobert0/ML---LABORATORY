import io
import tempfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import cross_val_score, train_test_split

from ml_lab.data import dataframe_to_bytes, demo_dataset, load_uploaded
from ml_lab.evaluation import classification_metrics, generalization_note, regression_metrics
from ml_lab.modeling import CLASSIFIERS, REGRESSORS, infer_task, make_pipeline

st.set_page_config(page_title="ML Lab", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1280px}.hero{padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#111827,#1f2937);color:white;margin-bottom:1rem}.hero h1{margin:0;font-size:2.2rem}.hero p{opacity:.85;margin:.4rem 0 0}.stMetric{border:1px solid rgba(128,128,128,.2);padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

for key, value in {"df": None, "target": None, "task": None, "features": [], "model": None, "split": None, "metrics": None}.items():
    st.session_state.setdefault(key, value)

PAGES = ["Início", "Dados e exploração", "Treinamento", "Predições", "Aprender"]
page = st.sidebar.radio("ML Lab", PAGES)
st.sidebar.caption("Laboratório educacional de Machine Learning")

def hero(title, text):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)

def set_dataset(df, target=None, task=None):
    st.session_state.df = df
    st.session_state.target = target or df.columns[-1]
    st.session_state.task = task or infer_task(df[st.session_state.target])
    st.session_state.features = [c for c in df.columns if c != st.session_state.target]
    st.session_state.model = None

if page == "Início":
    hero("ML Lab", "Importe, explore, treine, avalie e use modelos reais em um único laboratório.")
    st.subheader("Projetos demonstrativos")
    demos = ["Preços de imóveis (regressão)", "Câncer de mama (classificação)", "Progressão de diabetes (regressão)", "Satisfação populacional (regressão)"]
    cols = st.columns(4)
    descriptions = ["Estimativa de preço a partir de características do imóvel.", "Diagnóstico binário com atributos clínicos.", "Previsão de progressão a partir de variáveis clínicas.", "Relação entre indicadores socioeconômicos e satisfação."]
    for col, name, desc in zip(cols, demos, descriptions):
        with col:
            st.markdown(f"#### {name}")
            st.write(desc)
            if st.button("Usar projeto", key=name, use_container_width=True):
                set_dataset(*demo_dataset(name))
                st.success("Projeto carregado. Abra Dados e exploração.")
    st.divider()
    st.subheader("Experimento personalizado")
    upload = st.file_uploader("Importe CSV, TXT ou JSON", type=["csv", "txt", "json"], key="home_upload")
    sep = st.text_input("Separador para TXT", value=",")
    if upload and st.button("Importar dataset"):
        try:
            set_dataset(load_uploaded(upload, sep))
            st.success(f"Dataset importado com {len(st.session_state.df)} registros.")
        except Exception as exc:
            st.error(f"Não foi possível importar o arquivo. {exc}")

elif page == "Dados e exploração":
    hero("Dados e exploração", "Conheça o dataset antes de escolher como o modelo aprenderá.")
    df = st.session_state.df
    if df is None:
        st.info("Escolha um projeto ou importe um dataset na página Início.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", len(df)); c2.metric("Colunas", len(df.columns)); c3.metric("Ausentes", int(df.isna().sum().sum()))
    st.dataframe(df.head(100), use_container_width=True)
    target = st.selectbox("Variável-alvo", df.columns, index=list(df.columns).index(st.session_state.target) if st.session_state.target in df else len(df.columns)-1)
    available = [c for c in df.columns if c != target]
    features = st.multiselect("Variáveis preditoras", available, default=[c for c in st.session_state.features if c in available] or available)
    task_label = st.radio("Tipo do problema", ["Classificação", "Regressão"], index=0 if infer_task(df[target]) == "classification" else 1, horizontal=True)
    st.session_state.target, st.session_state.features = target, features
    st.session_state.task = "classification" if task_label == "Classificação" else "regression"
    tab1, tab2, tab3 = st.tabs(["Resumo", "Distribuições", "Correlação"])
    with tab1:
        summary = pd.DataFrame({"tipo": df.dtypes.astype(str), "ausentes": df.isna().sum(), "únicos": df.nunique(dropna=True)})
        st.dataframe(summary, use_container_width=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with tab2:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            col = st.selectbox("Atributo", numeric)
            fig, ax = plt.subplots(); sns.histplot(df[col].dropna(), kde=True, ax=ax); st.pyplot(fig); plt.close(fig)
    with tab3:
        corr = df.select_dtypes(include=np.number).corr()
        if corr.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(9, 6)); sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax); st.pyplot(fig); plt.close(fig)
        else: st.info("São necessárias ao menos duas colunas numéricas.")

elif page == "Treinamento":
    hero("Treinamento e avaliação", "Pré-processamento e modelo são treinados juntos para evitar vazamento de dados.")
    df, target, features, task = st.session_state.df, st.session_state.target, st.session_state.features, st.session_state.task
    if df is None or not target or not features:
        st.info("Configure dataset, alvo e preditores em Dados e exploração."); st.stop()
    clean = df.dropna(subset=[target]); X, y = clean[features], clean[target]
    a, b, c = st.columns(3)
    test_size = a.slider("Teste (%)", 10, 40, 20) / 100
    scaler = b.selectbox("Escalonamento", ["StandardScaler", "MinMaxScaler"])
    stratify = c.checkbox("Divisão estratificada", value=task == "classification", disabled=task != "classification")
    names = list(REGRESSORS if task == "regression" else CLASSIFIERS)
    model_name = st.selectbox("Modelo", names)
    params = {}
    if "Random Forest" == model_name:
        params["n_estimators"] = st.slider("Número de árvores", 50, 500, 150, 50)
        params["max_depth"] = st.slider("Profundidade máxima (0 = livre)", 0, 30, 0) or None
    elif "Árvore de Decisão" == model_name:
        params["max_depth"] = st.slider("Profundidade máxima", 1, 30, 6)
    elif model_name == "KNN": params["n_neighbors"] = st.slider("Vizinhos", 1, 25, 5)
    elif model_name in ["Ridge", "Lasso"]: params["alpha"] = st.number_input("Alpha", .001, 100.0, 1.0)
    elif model_name == "Regressão Logística": params["C"] = st.number_input("C (inverso da regularização)", .01, 100.0, 1.0)
    cv = st.slider("Folds da validação cruzada", 2, 10, 5)
    if st.button("Treinar modelo", type="primary", use_container_width=True):
        try:
            strat = y if stratify else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
            pipe = make_pipeline(Xtr, task, model_name, scaler, params); pipe.fit(Xtr, ytr)
            scoring = "r2" if task == "regression" else "accuracy"
            scores = cross_val_score(make_pipeline(X, task, model_name, scaler, params), X, y, cv=cv, scoring=scoring)
            st.session_state.model = pipe; st.session_state.split = (Xtr, Xte, ytr, yte); st.session_state.model_name = model_name
            if task == "regression": metrics, pred = regression_metrics(pipe, Xtr, ytr, Xte, yte); extra = None
            else: metrics, pred, cm, report = classification_metrics(pipe, Xtr, ytr, Xte, yte); extra = (cm, report)
            metrics["CV média"] = scores.mean(); metrics["CV desvio"] = scores.std(); st.session_state.metrics = metrics
            st.session_state.eval_data = (pred, extra)
        except Exception as exc: st.error(f"Falha no treinamento. {exc}")
    if st.session_state.model is not None and st.session_state.metrics:
        metrics = st.session_state.metrics
        cols = st.columns(min(5, len(metrics)))
        for col, (name, value) in zip(cols, metrics.items()): col.metric(name, f"{value:.4f}")
        st.info(generalization_note(task, metrics))
        Xtr, Xte, ytr, yte = st.session_state.split; pred, extra = st.session_state.eval_data
        if task == "regression":
            fig, ax = plt.subplots(); ax.scatter(yte, pred, alpha=.65); lo=min(yte.min(), pred.min()); hi=max(yte.max(), pred.max()); ax.plot([lo,hi],[lo,hi], "--"); ax.set(xlabel="Real", ylabel="Previsto", title="Real × previsto"); st.pyplot(fig); plt.close(fig)
        else:
            cm, report = extra; fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax); ax.set(xlabel="Previsto", ylabel="Real"); st.pyplot(fig); plt.close(fig); st.code(report)
        st.subheader("Comparar modelos")
        if st.button("Comparar todos"):
            rows=[]
            for name in names:
                try:
                    p=make_pipeline(Xtr, task, name, scaler); p.fit(Xtr,ytr)
                    m = regression_metrics(p,Xtr,ytr,Xte,yte)[0] if task=="regression" else classification_metrics(p,Xtr,ytr,Xte,yte)[0]
                    rows.append({"Modelo":name, **m})
                except Exception as exc: rows.append({"Modelo":name,"Erro":str(exc)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        bundle = {"pipeline": st.session_state.model, "features": features, "target": target, "task": task, "dtypes": X.dtypes.astype(str).to_dict(), "model_name": st.session_state.model_name}
        bio=io.BytesIO(); joblib.dump(bundle,bio)
        st.download_button("Baixar modelo (.joblib)", bio.getvalue(), "ml_lab_model.joblib", "application/octet-stream")

elif page == "Predições":
    hero("Predições", "Use o pipeline treinado em novos registros e exporte os resultados.")
    model = st.session_state.model
    uploaded_model = st.file_uploader("Ou carregue um modelo .joblib", type=["joblib"])
    bundle = None
    if uploaded_model:
        try: bundle = joblib.load(uploaded_model); model=bundle["pipeline"]
        except Exception as exc: st.error(f"Modelo inválido. {exc}")
    if model is None:
        st.info("Treine um modelo ou carregue um arquivo .joblib."); st.stop()
    if bundle:
        features, task = bundle["features"], bundle["task"]
        saved_dtypes = bundle.get("dtypes", {})
    else:
        features, task = st.session_state.features, st.session_state.task
        saved_dtypes = {}
    tab_manual, tab_batch = st.tabs(["Entrada manual", "Importar registros"])
    with tab_manual:
        values={}; source=st.session_state.df
        cols=st.columns(2)
        for i, f in enumerate(features):
            with cols[i%2]:
                if source is not None and f in source and pd.api.types.is_numeric_dtype(source[f]): values[f]=st.number_input(f, value=float(source[f].median()) if source[f].notna().any() else 0.0)
                elif source is not None and f in source:
                    opts=source[f].dropna().astype(str).unique().tolist(); values[f]=st.selectbox(f, opts) if opts else st.text_input(f)
                elif saved_dtypes.get(f, "").startswith(("int", "float")): values[f]=st.number_input(f, value=0.0)
                else: values[f]=st.text_input(f)
        if st.button("Prever", type="primary"):
            row=pd.DataFrame([values]); result=model.predict(row)[0]; st.success(f"Resultado previsto: {result}")
            if task=="classification" and hasattr(model,"predict_proba"):
                probs=model.predict_proba(row)[0]; classes=model.classes_; st.dataframe(pd.DataFrame({"Classe":classes,"Probabilidade":probs}), hide_index=True)
    with tab_batch:
        batch=st.file_uploader("CSV, TXT ou JSON com os preditores", type=["csv","txt","json"], key="batch")
        if batch:
            try:
                new=load_uploaded(batch); missing=[f for f in features if f not in new]
                if missing: st.error("Colunas ausentes: "+", ".join(missing))
                else:
                    out=new.copy(); out["predicao"]=model.predict(new[features])
                    if task=="classification" and hasattr(model,"predict_proba"):
                        for cls, vals in zip(model.classes_, model.predict_proba(new[features]).T): out[f"prob_{cls}"]=vals
                    st.dataframe(out, use_container_width=True)
                    fmt=st.radio("Formato",["CSV","JSON","TXT"],horizontal=True); ext=fmt.lower(); st.download_button("Exportar resultados", dataframe_to_bytes(out,fmt), f"predicoes.{ext}")
            except Exception as exc: st.error(f"Falha na predição. {exc}")

else:
    hero("Aprender Machine Learning", "Conceitos do laboratório conectados ao que você observa nos experimentos.")
    topics = {
        "Aprendizado supervisionado": "O modelo aprende a relação entre atributos (X) e uma resposta conhecida (y). Regressão prevê valores numéricos e classificação prevê classes.",
        "Parâmetros e hiperparâmetros": "Parâmetros são aprendidos durante o treino. Hiperparâmetros, como profundidade da árvore, C ou alpha, são definidos antes e controlam o aprendizado.",
        "Função de custo e métricas": "A função de custo orienta a otimização. Métricas como RMSE, R², acurácia e F1 avaliam o resultado com uma interpretação adequada ao problema.",
        "Gradiente descendente": "Atualiza parâmetros na direção que reduz o custo. A taxa de aprendizado controla o tamanho de cada passo. Escalonar atributos costuma facilitar a otimização.",
        "Overfitting, underfitting e generalização": "Overfitting aparece quando o modelo aprende demais o treino e perde desempenho fora dele. Underfitting ocorre quando o modelo é simples demais. Compare treino, teste e validação cruzada.",
        "Viés e variância": "Alto viés costuma acompanhar underfitting. Alta variância costuma acompanhar overfitting. O objetivo prático é equilibrar ambos para generalizar.",
        "Regularização L1 e L2": "Lasso (L1) pode zerar coeficientes. Ridge (L2) reduz pesos grandes. Alpha controla a força da penalização. Na regressão logística, C funciona de forma inversa à regularização.",
        "Árvores, CART, Gini e entropia": "CART escolhe divisões que tornam os nós mais puros. Gini e entropia medem impureza em classificação. Limitar profundidade ajuda a controlar variância.",
        "Bagging, bootstrap e OOB": "Random Forest treina árvores em amostras bootstrap e agrega previsões. Registros fora de cada amostra formam o conjunto Out-of-Bag daquela árvore e podem estimar generalização.",
    }
    for title, text in topics.items():
        with st.expander(title): st.write(text)
    if st.session_state.metrics:
        st.subheader("Ligação com seu último experimento")
        st.write(generalization_note(st.session_state.task, st.session_state.metrics))

