import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.text_preprocessing import preprocess_dataframe, clean_text
from src.models import train_models
from src.similarity import TicketSimilarityEngine
from src.evaluation import classification_metrics


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="IT Support Ticket Intelligence",
    page_icon="",
    layout="wide"
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/it_support_tickets.csv"
    )

    df = preprocess_dataframe(
        df,
        "ticket_text"
    )

    return df


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

@st.cache_resource
def build_system(df):

    X = df["clean_text"]

    y_category = df["category"]

    y_issue = df["issue_type"]

    y_priority = df["priority"]


    (
        X_train,
        X_test,
        category_train,
        category_test,
        issue_train,
        issue_test,
        priority_train,
        priority_test
    ) = train_test_split(

        X,
        y_category,
        y_issue,
        y_priority,

        test_size=0.25,

        random_state=42,

        stratify=y_category
    )


    # Train three models
    (
        category_model,
        issue_model,
        priority_model
    ) = train_models(

        X_train,

        category_train,

        issue_train,

        priority_train
    )


    # Predictions
    category_predictions = category_model.predict(
        X_test
    )

    issue_predictions = issue_model.predict(
        X_test
    )

    priority_predictions = priority_model.predict(
        X_test
    )


    # Evaluation
    metrics = {

        "Category":
            classification_metrics(
                category_test,
                category_predictions
            ),

        "Issue Type":
            classification_metrics(
                issue_test,
                issue_predictions
            ),

        "Priority":
            classification_metrics(
                priority_test,
                priority_predictions
            )
    }


    # Similarity engine
    similarity_engine = TicketSimilarityEngine(
        df["clean_text"]
    )


    return {

        "category_model": category_model,

        "issue_model": issue_model,

        "priority_model": priority_model,

        "category_test": category_test,

        "issue_test": issue_test,

        "priority_test": priority_test,

        "category_predictions":
            category_predictions,

        "issue_predictions":
            issue_predictions,

        "priority_predictions":
            priority_predictions,

        "metrics":
            metrics,

        "similarity":
            similarity_engine
    }


# ---------------------------------------------------------
# LOAD SYSTEM
# ---------------------------------------------------------

df = load_data()

system = build_system(df)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title(
    " IT Support Ticket Intelligence & Automated Routing"
)

st.caption(
    "NLP-based ticket classification, priority prediction, "
    "automated routing and historical-ticket retrieval."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title(
    "Dataset Overview"
)

st.sidebar.metric(
    "Historical Tickets",
    len(df)
)

st.sidebar.metric(
    "Categories",
    df["category"].nunique()
)

st.sidebar.metric(
    "Issue Types",
    df["issue_type"].nunique()
)

st.sidebar.metric(
    "Support Teams",
    df["assigned_team"].nunique()
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        " Ticket Analyzer",
        " Model Evaluation",
        " Historical Tickets",
        " Methodology"
    ]
)


# =========================================================
# TAB 1 — TICKET ANALYZER
# =========================================================

with tab1:

    st.subheader(
        "Analyze an Incoming IT Support Ticket"
    )


    ticket = st.text_area(

        "Enter the support request",

        value=(
            "My laptop suddenly became extremely slow "
            "after the latest Windows update. "
            "I have a client presentation in 2 hours."
        ),

        height=130
    )


    analyze = st.button(
        "Analyze Ticket",
        type="primary"
    )


    if analyze:

        if not ticket.strip():

            st.warning(
                "Please enter a support ticket."
            )

            st.stop()


        # -----------------------------
        # PREPROCESS TEXT
        # -----------------------------

        cleaned_text = clean_text(
            ticket
        )


        # -----------------------------
        # MODELS
        # -----------------------------

        category_model = system[
            "category_model"
        ]

        issue_model = system[
            "issue_model"
        ]

        priority_model = system[
            "priority_model"
        ]


        # -----------------------------
        # PREDICTIONS
        # -----------------------------

        category = category_model.predict(
            [cleaned_text]
        )[0]

        issue_type = issue_model.predict(
            [cleaned_text]
        )[0]

        priority = priority_model.predict(
            [cleaned_text]
        )[0]


        # -----------------------------
        # CONFIDENCE
        # -----------------------------

        category_confidence = np.max(
            category_model.predict_proba(
                [cleaned_text]
            )
        )

        issue_confidence = np.max(
            issue_model.predict_proba(
                [cleaned_text]
            )
        )

        priority_confidence = np.max(
            priority_model.predict_proba(
                [cleaned_text]
            )
        )


        overall_confidence = np.mean(
            [
                category_confidence,
                issue_confidence,
                priority_confidence
            ]
        )


        # -----------------------------
        # ROUTING
        # -----------------------------

        routing_rules = {

            "Account & Access":
                "Identity & Access",

            "Hardware":
                "Desktop Support",

            "Network":
                "Network Support",

            "Email":
                "Email Support",

            "Security":
                "Security Team",

            "Software":
                "Application Support"
        }


        assigned_team = routing_rules.get(
            category,
            "General IT Support"
        )


        # -----------------------------
        # SENTIMENT SIGNAL
        # -----------------------------

        negative_words = {

            "cannot",
            "cant",
            "failed",
            "failure",
            "crash",
            "crashes",
            "slow",
            "down",
            "suspicious",
            "malware",
            "flickering",
            "error",
            "locked",
            "urgent",
            "broken"
        }


        words = set(
            cleaned_text.split()
        )


        if negative_words.intersection(words):

            sentiment = "Negative"

        else:

            sentiment = "Neutral"


        # -----------------------------
        # DISPLAY PREDICTIONS
        # -----------------------------

        st.markdown(
            "### Structured Ticket Analysis"
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Category",
            category
        )

        col2.metric(
            "Issue Type",
            issue_type
        )

        col3.metric(
            "Priority",
            priority
        )

        col4.metric(
            "Confidence",
            f"{overall_confidence * 100:.1f}%"
        )


        st.info(
            f"**Assigned Team:** {assigned_team}  \n"
            f"**Sentiment Signal:** {sentiment}"
        )


        # -----------------------------
        # SIMILAR TICKETS
        # -----------------------------

        st.markdown(
            "### Similar Historical Tickets"
        )


        indices, scores = (
            system["similarity"].find_similar(
                cleaned_text,
                top_k=5
            )
        )


        similar_tickets = df.iloc[
            indices
        ][
            [
                "ticket_id",
                "ticket_text",
                "category",
                "issue_type",
                "priority",
                "assigned_team",
                "resolution"
            ]
        ].copy()


        similar_tickets[
            "Similarity"
        ] = [

            f"{score * 100:.1f}%"

            for score in scores
        ]


        st.dataframe(
            similar_tickets,
            use_container_width=True,
            hide_index=True
        )


        # -----------------------------
        # RESOLUTION
        # -----------------------------

        st.markdown(
            "### Suggested Resolution"
        )


        best_resolution = df.iloc[
            indices[0]
        ]["resolution"]


        st.success(
            best_resolution
        )


        st.caption(
            "This suggestion is retrieved from the "
            "synthetic historical dataset. It is not "
            "a guaranteed resolution."
        )


# =========================================================
# TAB 2 — MODEL EVALUATION
# =========================================================

with tab2:

    st.subheader(
        "Model Performance Evaluation"
    )


    metric_rows = []


    for task, values in system[
        "metrics"
    ].items():

        metric_rows.append(

            {
                "Task": task,

                "Accuracy":
                    round(
                        values["Accuracy"],
                        4
                    ),

                "Precision":
                    round(
                        values["Precision"],
                        4
                    ),

                "Recall":
                    round(
                        values["Recall"],
                        4
                    ),

                "F1 Score":
                    round(
                        values["F1 Score"],
                        4
                    )
            }
        )


    metrics_df = pd.DataFrame(
        metric_rows
    )


    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### Confusion Matrices"
    )


    evaluation_tasks = [

        (
            "Category",

            system["category_test"],

            system["category_predictions"]
        ),

        (
            "Issue Type",

            system["issue_test"],

            system["issue_predictions"]
        ),

        (
            "Priority",

            system["priority_test"],

            system["priority_predictions"]
        )
    ]


    for title, actual, predicted in evaluation_tasks:

        labels = sorted(
            pd.Series(
                actual
            ).unique()
        )


        matrix = confusion_matrix(
            actual,
            predicted,
            labels=labels
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )


        sns.heatmap(

            matrix,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=labels,

            yticklabels=labels,

            ax=ax
        )


        ax.set_title(
            f"{title} Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.yticks(
            rotation=0
        )


        st.pyplot(
            fig
        )


        plt.close(fig)


# =========================================================
# TAB 3 — HISTORICAL DATA
# =========================================================

with tab3:

    st.subheader(
        "Synthetic Historical Ticket Dataset"
    )


    display_df = df.drop(
        columns=["clean_text"]
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    csv_data = display_df.to_csv(
        index=False
    )


    st.download_button(

        "Download Dataset CSV",

        data=csv_data,

        file_name="it_support_tickets.csv",

        mime="text/csv"
    )


# =========================================================
# TAB 4 — METHODOLOGY
# =========================================================

with tab4:

    st.subheader(
        "System Methodology"
    )


    st.markdown(
        """
### NLP Pipeline

**Incoming Ticket**

↓

**Text Preprocessing**

↓

**TF-IDF Feature Extraction**

↓

**Three Classification Models**

- Category prediction
- Issue-type prediction
- Priority prediction

↓

**Automated Team Routing**

↓

**Historical Ticket Similarity**

↓

**Resolution Suggestion**

### Text Preprocessing

The system:

- converts text to lowercase
- normalizes Wi-Fi/e-mail variations
- expands selected technical abbreviations
- removes unnecessary special characters
- normalizes whitespace

### TF-IDF

TF-IDF converts ticket text into numerical features based on
how important words and word combinations are within the dataset.

The project uses both:

- unigrams
- bigrams

This allows the model to learn phrases such as:

`password reset`

`VPN connection`

`application crash`

`wireless network`

### Classification

Logistic Regression is used for the three prediction tasks.

### Similarity Search

Historical tickets are converted into TF-IDF vectors.

Cosine similarity is then used to identify tickets that are
textually similar to the incoming request.

### Routing

Routing is based on the predicted category.

The routing rules are project-defined demonstration rules,
not real enterprise escalation policies.

### Priority

Priority labels are project-defined training labels and should
not be interpreted as actual company SLA commitments.
"""
    )