import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

st.set_page_config(page_title="Diabeta AI", page_icon="🩺", layout="wide")

FEATURES = ["Pregnancies","Glucose","BloodPressure","Insulin","BMI","DiabetesPedigreeFunction","Age"]
TARGET = "Outcome"
DATA_PATH = "data/diabetes.csv"


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], [class*="st-"], span, p, div {
    font-family: 'Inter', sans-serif !important;
}

/* Main Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    color: #1e293b;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 100%);
    box-shadow: 2px 0 15px rgba(186, 230, 253, 0.4);
}
[data-testid="stSidebar"] * {color: #1e3a8a !important;}

/* Increase Sidebar Text Size */
[data-testid="stSidebar"] .stRadio label p, 
[data-testid="stSidebar"] .stRadio div, 
[data-testid="stSidebar"] label span {
    font-size: 20px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 18px !important;
}

/* Brand and Titles */
.brand {padding: 15px 4px 30px; text-align: center;}
.brand-title {
    font-size: 38px !important; font-weight: 800; 
    background: linear-gradient(90deg, #1e40af 0%, #0369a1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {color: #3b82f6 !important; font-size: 16px !important; margin-top: 5px; font-weight: 600;}

/* Hero Banner */
.hero {
    padding: 40px; border-radius: 24px; 
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    border: 2px solid rgba(255,255,255,0.6);
    box-shadow: 0 10px 30px rgba(253, 160, 133, 0.4);
    color: #5a2e15; margin-bottom: 30px;
    text-align: center;
}
.hero h1 {
    margin: 0; font-size: 42px; font-weight: 800;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
    color: #7c2d12;
}
.hero p {margin: 15px 0 0; color: #9a3412; font-size: 18px; font-weight: 500;}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.85);
    border: 2px solid rgba(255, 255, 255, 1);
    border-radius: 20px; padding: 25px;
    box-shadow: 0 8px 25px 0 rgba(148, 163, 184, 0.2);
    backdrop-filter: blur(8px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px 0 rgba(148, 163, 184, 0.4);
    border: 2px solid #60a5fa;
}
.card-title {font-size: 14px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;}
.card-value {
    font-size: 36px; font-weight: 800; margin: 10px 0;
    background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card-note {font-size: 13px; color: #94a3b8; font-weight: 500;}

/* Result Boxes */
.result-positive {
    padding: 30px; border-radius: 20px;
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #fca5a5;
    box-shadow: 0 5px 20px rgba(239, 68, 68, 0.15);
}
.result-negative {
    padding: 30px; border-radius: 20px;
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #86efac;
    box-shadow: 0 5px 20px rgba(16, 185, 129, 0.15);
}
.result-title {font-size: 28px; font-weight: 800; margin-top: 10px; color: #0f172a;}
.small-muted {color: #475569; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;}

/* Headings and Sections */
.section {
    font-size: 24px; font-weight: 800; color: #1e293b;
    margin: 20px 0 20px;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 10px;
}

/* Base Elements */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.9);
    border: 2px solid white;
    padding: 20px; border-radius: 16px;
    box-shadow: 0 5px 15px rgba(148, 163, 184, 0.2);
}
.stButton>button {
    border-radius: 12px; font-weight: 800; font-size: 16px;
    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
    border: none; color: white;
    padding: 10px 24px; transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6);
}

/* Other Tweaks */
p, li {color: #334155; font-size: 16px; line-height: 1.6; font-weight: 500;}
footer {visibility: hidden;}

/* Inputs */
.stNumberInput > div > div > input {
    background-color: white !important;
    color: #0f172a !important;
    border-radius: 8px;
    border: 2px solid #cbd5e1 !important;
    font-weight: 600;
}
.stNumberInput > div > div > input:focus {
    border-color: #3b82f6 !important;
}

/* Fix text in graphs for light theme */
.js-plotly-plot .plotly .main-svg text { fill: #1e293b !important; }

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_models(df):
    X=df[FEATURES].copy(); y=df[TARGET].astype(int)
    missing=["Glucose","BloodPressure","Insulin","BMI"]
    X[missing]=X[missing].replace(0,np.nan)
    medians=X.median(); X=X.fillna(medians)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")
    xgb=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=42)
    rf.fit(Xtr,ytr); xgb.fit(Xtr,ytr)
    return rf,xgb,Xte,yte,medians

def metrics(model,X,y):
    p=model.predict(X); prob=model.predict_proba(X)[:,1]
    return {"Accuracy":accuracy_score(y,p),"Precision":precision_score(y,p,zero_division=0),"Recall":recall_score(y,p,zero_division=0),"F1 Score":f1_score(y,p,zero_division=0),"ROC-AUC":roc_auc_score(y,prob)}

df=load_data(); rf,xgb,X_test,y_test,medians=train_models(df)
rfm=metrics(rf,X_test,y_test); xgbm=metrics(xgb,X_test,y_test)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-title">🩺 Diabeta AI</div><div class="brand-sub">Diabetes Prediction & Explainability</div></div>',unsafe_allow_html=True)
    page=st.radio("NAVIGATION",["Dashboard","Prediction","Model Analytics","Feature Importance","Dataset Explorer","About"])
    st.markdown("---"); st.caption("ML Research Project"); st.caption("Random Forest + XGBoost")

if page=="Dashboard":
    st.markdown('<div class="hero"><h1>Diabetes Intelligence Dashboard</h1><p>Compare predictive performance and understand which clinical features influence model decisions.</p></div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    cards=[("PATIENT RECORDS",f"{len(df):,}","Pima-style dataset"),("FEATURES",str(len(FEATURES)),"Clinical variables"),("RF ROC-AUC",f"{rfm['ROC-AUC']:.3f}","Random Forest"),("XGB ROC-AUC",f"{xgbm['ROC-AUC']:.3f}","XGBoost")]
    for c,(t,v,n) in zip([c1,c2,c3,c4],cards):
        with c: st.markdown(f'<div class="card"><div class="card-title">{t}</div><div class="card-value">{v}</div><div class="card-note">{n}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        out=pd.DataFrame({"Outcome":["No Diabetes","Diabetes"],"Patients":[int((df.Outcome==0).sum()),int((df.Outcome==1).sum())]})
        fig=px.pie(out,names="Outcome",values="Patients",hole=.58); fig.update_layout(height=340,margin=dict(t=10,b=10,l=10,r=10))
        st.markdown('<div class="section">Outcome Distribution</div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True)
    with b:
        perf=pd.DataFrame({"Metric":list(rfm),"Random Forest":list(rfm.values()),"XGBoost":list(xgbm.values())})
        fig=px.bar(perf,x="Metric",y=["Random Forest","XGBoost"],barmode="group",range_y=[0,1]); fig.update_layout(height=340,margin=dict(t=10,b=10,l=10,r=10))
        st.markdown('<div class="section">Model Performance</div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section">Project Workflow</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><b>DATA</b> → <b>PREPROCESSING</b> → <b>TRAIN</b> → <b>PREDICT</b> → <b>COMPARE</b> → <b>EXPLAIN</b></div>',unsafe_allow_html=True)

elif page=="Prediction":
    st.markdown('<div class="hero"><h1>Patient Risk Prediction</h1><p>Enter clinical measurements and compare both trained models.</p></div>',unsafe_allow_html=True)
    with st.form("prediction"):
        st.markdown('<div class="section">Patient Information</div>',unsafe_allow_html=True)
        a,b,c=st.columns(3)
        with a: preg=st.number_input("Pregnancies",0,20,1); glucose=st.number_input("Glucose",0,250,120); bp=st.number_input("Blood Pressure",0,150,70)
        with b: insulin=st.number_input("Insulin",0,900,80); bmi=st.number_input("BMI",0.,70.,30.)
        with c: dpf=st.number_input("Diabetes Pedigree Function",0.,3.,.50,.01); age=st.number_input("Age",1,120,30); st.write(""); submit=st.form_submit_button("🔍 ANALYZE RISK",use_container_width=True)
    if submit:
        v=pd.DataFrame([[preg,glucose,bp,insulin,bmi,dpf,age]],columns=FEATURES)
        missing=["Glucose","BloodPressure","Insulin","BMI"]; v[missing]=v[missing].replace(0,np.nan); v=v.fillna(medians)
        rp=int(rf.predict(v)[0]); rprob=float(rf.predict_proba(v)[0,1]); xp=int(xgb.predict(v)[0]); xprob=float(xgb.predict_proba(v)[0,1])
        a,b=st.columns(2)
        for c,name,p,prob in [(a,"Random Forest",rp,rprob),(b,"XGBoost",xp,xprob)]:
            with c:
                cls="result-positive" if p else "result-negative"; label="Higher predicted risk" if p else "Lower predicted risk"
                st.markdown(f'<div class="{cls}"><div class="small-muted">{name}</div><div class="result-title">{label}</div><p>Predicted probability: <b>{prob:.1%}</b></p></div>',unsafe_allow_html=True)
        st.warning("Educational/research project only. This is not a medical diagnosis.")

elif page=="Model Analytics":
    st.markdown('<div class="hero"><h1>Model Analytics</h1><p>Evaluate both models on the same held-out test set.</p></div>',unsafe_allow_html=True)
    perf=pd.DataFrame({"Metric":list(rfm),"Random Forest":list(rfm.values()),"XGBoost":list(xgbm.values())})
    st.dataframe(perf.style.format({"Random Forest":"{:.3f}","XGBoost":"{:.3f}"}),use_container_width=True,hide_index=True)
    a,b=st.columns(2)
    for c,m,name in [(a,rf,"Random Forest"),(b,xgb,"XGBoost")]:
        with c:
            cm=confusion_matrix(y_test,m.predict(X_test)); fig=px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual"),x=["No Diabetes","Diabetes"],y=["No Diabetes","Diabetes"]); fig.update_layout(height=380)
            st.markdown(f'<div class="section">{name} Confusion Matrix</div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True)
    rprob=rf.predict_proba(X_test)[:,1]; xprob=xgb.predict_proba(X_test)[:,1]; rf_f,rf_t,_=roc_curve(y_test,rprob); x_f,x_t,_=roc_curve(y_test,xprob)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=rf_f,y=rf_t,mode="lines",name=f"Random Forest (AUC {rfm['ROC-AUC']:.3f})")); fig.add_trace(go.Scatter(x=x_f,y=x_t,mode="lines",name=f"XGBoost (AUC {xgbm['ROC-AUC']:.3f})")); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Baseline",line=dict(dash="dash"))); fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",height=450)
    st.markdown('<div class="section">ROC Curve</div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True)

elif page=="Feature Importance":
    st.markdown('<div class="hero"><h1>Feature Importance & Explainability</h1><p>Compare which variables the trained models rely on most.</p></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    rfi=pd.Series(rf.feature_importances_,index=FEATURES).sort_values(); xbi=pd.Series(xgb.feature_importances_,index=FEATURES).sort_values()
    with a:
        fig=px.bar(x=rfi.values,y=rfi.index,orientation="h",title="Random Forest — Gini Importance",labels={"x":"Importance","y":"Feature"}); fig.update_layout(height=430); st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.bar(x=xbi.values,y=xbi.index,orientation="h",title="XGBoost — Feature Importance",labels={"x":"Importance","y":"Feature"}); fig.update_layout(height=430); st.plotly_chart(fig,use_container_width=True)
    table=pd.DataFrame({"Feature":FEATURES,"Random Forest":rf.feature_importances_,"XGBoost":xgb.feature_importances_})
    table["RF Rank"]=table["Random Forest"].rank(ascending=False,method="min").astype(int); table["XGB Rank"]=table["XGBoost"].rank(ascending=False,method="min").astype(int)
    st.dataframe(table.sort_values("RF Rank").style.format({"Random Forest":"{:.4f}","XGBoost":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.info("Feature importance indicates model reliance; it does not prove that a feature causes diabetes.")

elif page=="Dataset Explorer":
    st.markdown('<div class="hero"><h1>Dataset Explorer</h1><p>Explore the exact diabetes dataset used by the application.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Rows",df.shape[0]); b.metric("Columns",df.shape[1]); c.metric("Missing Cells",int(df.isna().sum().sum()))
    st.markdown('<div class="section">Dataset Preview</div>',unsafe_allow_html=True); st.dataframe(df.head(25),use_container_width=True,hide_index=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.histogram(df,x="Glucose",color="Outcome",nbins=30,barmode="overlay"),use_container_width=True)
    with b: st.plotly_chart(px.scatter(df,x="BMI",y="Glucose",color="Outcome",hover_data=["Age","Pregnancies"]),use_container_width=True)
    st.markdown('<div class="section">Descriptive Statistics</div>',unsafe_allow_html=True); st.dataframe(df.describe().T,use_container_width=True)

else:
    st.markdown('<div class="hero"><h1>About the Project</h1><p>Feature Importance Analysis in Diabetes Prediction Using Random Forest and XGBoost</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Objective</h3><p>Build two machine-learning classifiers for diabetes prediction and compare their performance and feature importance.</p><h3>Models</h3><ul><li><b>Random Forest:</b> combines many decision trees using bagging.</li><li><b>XGBoost:</b> builds boosted trees sequentially to reduce errors.</li></ul><h3>Technology</h3><p>Python, Streamlit, Pandas, NumPy, Scikit-learn, XGBoost and Plotly.</p></div>',unsafe_allow_html=True)
    st.warning("Academic/research demonstration only; not a medical diagnosis tool.")