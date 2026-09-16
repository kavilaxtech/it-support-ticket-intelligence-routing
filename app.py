import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from src.text_preprocessing import clean_text
from src.models import train_models
from src.similarity import TicketSimilarityEngine
from src.evaluation import classification_metrics

st.set_page_config(page_title="IT Support Ticket Intelligence", page_icon="🎫", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/it_support_tickets.csv")
    df["clean_text"] = df["ticket_text"].apply(clean_text)
    return df

@st.cache_resource
def build_system(df):
    X = df["clean_text"]
    yc, yi, yp = df["category"], df["issue_type"], df["priority"]
    Xtr, Xte, yc_tr, yc_te, yi_tr, yi_te, yp_tr, yp_te = train_test_split(
        X, yc, yi, yp, test_size=0.25, random_state=42, stratify=yc
    )
    cm, im, pm = train_models(Xtr, yc_tr, yi_tr, yp_tr)
    pc, pi, pp = cm.predict(Xte), im.predict(Xte), pm.predict(Xte)
    metrics = {
        "Category": classification_metrics(yc_te, pc),
        "Issue Type": classification_metrics(yi_te, pi),
        "Priority": classification_metrics(yp_te, pp),
    }
    return dict(category=cm, issue=im, priority=pm, yc=yc_te, yi=yi_te, yp=yp_te,
                pc=pc, pi=pi, pp=pp, metrics=metrics,
                similarity=TicketSimilarityEngine(df["clean_text"]))

df = load_data()
system = build_system(df)

st.title("🎫 IT Support Ticket Intelligence & Automated Routing")
st.caption("NLP-based ticket classification, priority assessment, routing and historical-ticket retrieval.")

tab1, tab2, tab3, tab4 = st.tabs(["Ticket Analyzer","Model Evaluation","Historical Tickets","Methodology"])

with tab1:
    st.subheader("Analyze an Incoming Ticket")
    ticket = st.text_area("Enter the IT support request",
        "My laptop suddenly became extremely slow after the latest Windows update. I have a client presentation in 2 hours.",
        height=120)
    if st.button("Analyze Ticket", type="primary"):
        text = clean_text(ticket)
        cm, im, pm = system["category"], system["issue"], system["priority"]
        category, issue, priority = cm.predict([text])[0], im.predict([text])[0], pm.predict([text])[0]
        confidence = np.mean([cm.predict_proba([text]).max(), im.predict_proba([text]).max(), pm.predict_proba([text]).max()])
        routing = {"Account & Access":"Identity & Access","Hardware":"Desktop Support","Network":"Network Support",
                   "Email":"Email Support","Security":"Security Team","Software":"Application Support"}
        team = routing.get(category,"General IT Support")
        negative_words = {"cannot","cant","failed","failure","crash","crashes","slow","down","suspicious","malware","flickering","error","locked","urgent","broken"}
        sentiment = "negative" if negative_words.intersection(text.split()) else "neutral"
        idx, scores = system["similarity"].find_similar(text, 5)
        similar = df.iloc[idx][["ticket_id","ticket_text","category","issue_type","priority","assigned_team","resolution"]].copy()
        similar["Similarity"] = [f"{x*100:.1f}%" for x in scores]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Category",category); c2.metric("Issue Type",issue)
        c3.metric("Priority",priority); c4.metric("Confidence",f"{confidence*100:.1f}%")
        st.info(f"**Assigned Team:** {team}  |  **Sentiment Signal:** {sentiment}")
        st.markdown("### Similar Historical Tickets")
        st.dataframe(similar,use_container_width=True,hide_index=True)
        st.markdown("### Suggested Resolution")
        st.success(df.iloc[idx[0]]["resolution"])
        st.caption("Retrieved from the synthetic project dataset; this is a suggestion, not a guaranteed resolution.")

with tab2:
    st.subheader("Model Performance")
    rows = [{"Task":k,**{m:round(v,4) for m,v in vals.items()}} for k,vals in system["metrics"].items()]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    for title, yt, yp in [("Category",system["yc"],system["pc"]),("Issue Type",system["yi"],system["pi"]),("Priority",system["yp"],system["pp"])]:
        labels=sorted(pd.Series(yt).unique())
        cmx=confusion_matrix(yt,yp,labels=labels)
        fig,ax=plt.subplots(figsize=(8,5))
        sns.heatmap(cmx,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
        ax.set_title(f"{title} Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.xticks(rotation=35,ha="right"); plt.yticks(rotation=0); st.pyplot(fig); plt.close(fig)

with tab3:
    st.subheader("Synthetic Historical Tickets")
    st.dataframe(df.drop(columns=["clean_text"]),use_container_width=True,hide_index=True)
    st.download_button("Download Dataset CSV",df.drop(columns=["clean_text"]).to_csv(index=False),
                       "it_support_tickets.csv","text/csv")

with tab4:
    st.subheader("Methodology")
    st.markdown("""**Pipeline:** ticket text -> cleaning -> TF-IDF unigram/bigram features -> separate Logistic Regression classifiers -> category/issue/priority -> deterministic routing -> cosine-similarity retrieval -> resolution suggestion.

**Priority and routing:** project-defined demonstration assumptions, not real enterprise SLA or escalation policy.

**Similarity:** cosine similarity between TF-IDF vectors; higher similarity indicates greater textual overlap within this synthetic dataset.
""")

st.sidebar.header("Dataset Overview")
st.sidebar.metric("Historical Tickets",len(df))
st.sidebar.metric("Categories",df.category.nunique())
st.sidebar.metric("Issue Types",df.issue_type.nunique())
st.sidebar.metric("Teams",df.assigned_team.nunique())
