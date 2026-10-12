"""
Real-Time Fraud Detection System - Streamlit Interactive Dashboard
Features:
- Overview KPI metrics and executive summary
- Real-Time Transaction Scoring workbench with one-click authentic sample presets
- Bulk CSV file upload, batch inference, and exportable predictions
- Deep Fraud Analytics with interactive Plotly visualisations
- Model Performance comparison (PR-AUC, ROC-AUC, Confusion Matrix, Precision/Recall)
- Transaction Audit History with search, risk filters, and inspection modal
"""

import os
import io
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database
from predict import get_predictor, FraudPredictor
from preprocessing import NUMERICAL_FEATURES, PCA_FEATURES

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="FinShield AI | Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Financial Analytics UI
st.markdown("""
<style>
    /* Global Styles */
    .main { background-color: #0b0f19; }
    h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #161e2e 0%, #1a2436 100%);
        border: 1px solid #283548;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Risk Badges */
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_predictor():
    """Initializes and caches the fraud prediction service."""
    database.init_db()
    return get_predictor()


@st.cache_data
def load_sample_presets():
    """Loads sample transactions for convenient dashboard demonstrations."""
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_transactions.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return pd.DataFrame()


@st.cache_data
def load_model_evaluation_metrics():
    """Loads model training comparison and curves JSON."""
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


predictor = load_system_predictor()
sample_df = load_sample_presets()
metrics_data = load_model_evaluation_metrics()


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FinShield AI")
    st.caption("Real-Time Fraud Prevention & Risk Intelligence")
    st.divider()

    navigation = st.radio(
        "Navigation Menu",
        [
            "📊 Executive Overview",
            "⚡ Real-Time Detection",
            "📁 Bulk CSV Analysis",
            "📈 Fraud Analytics",
            "🧠 Model Performance",
            "📜 Transaction Audit History"
        ]
    )

    st.divider()
    st.subheader("⚙️ Risk Thresholds")
    low_thresh = st.slider("Low-Risk Cutoff", 0.05, 0.40, float(predictor.threshold_low), 0.05)
    high_thresh = st.slider("High-Risk Cutoff", 0.45, 0.90, float(predictor.threshold_high), 0.05)
    
    # Update runtime thresholds
    predictor.threshold_low = low_thresh
    predictor.threshold_high = high_thresh

    st.caption(f"Low: < {low_thresh*100:.0f}% | Medium: {low_thresh*100:.0f}%–{high_thresh*100:.0f}% | High: ≥ {high_thresh*100:.0f}%")
    st.divider()
    st.info("💡 **Prototype Notice**: Educational system trained on European transaction benchmark.")


# -------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if navigation == "📊 Executive Overview":
    st.title("📊 Executive Fraud Operations Overview")
    st.markdown("Real-time summary of transaction volume, flagged anomalies, and risk metrics.")

    stats = database.get_transaction_stats()

    # KPI Top Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Processed</div>
            <div class="metric-value">{stats['total_transactions']:,}</div>
            <div class="metric-sub">Total transaction volume: €{stats['total_volume_amount']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Flagged for Review</div>
            <div class="metric-value" style="color: #f59e0b;">{stats['flagged_transactions']:,}</div>
            <div class="metric-sub">High & Medium risk alerts</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Alert Rate</div>
            <div class="metric-value" style="color: {'#ef4444' if stats['fraud_rate_percentage'] > 1.0 else '#10b981'};">
                {stats['fraud_rate_percentage']:.2f}%
            </div>
            <div class="metric-sub">Baseline normal: ~0.17%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">At-Risk Capital</div>
            <div class="metric-value" style="color: #ef4444;">€{stats['flagged_volume_amount']:,.2f}</div>
            <div class="metric-sub">Protected value in flagged txns</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Visual Breakdown Columns
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("Risk Category Distribution")
        risk_counts = {
            "Low Risk (Approved)": stats["low_risk_count"],
            "Medium Risk (2FA / Review)": stats["medium_risk_count"],
            "High Risk (Blocked)": stats["high_risk_count"]
        }
        fig_donut = px.pie(
            names=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            hole=0.6,
            color=list(risk_counts.keys()),
            color_discrete_map={
                "Low Risk (Approved)": "#10b981",
                "Medium Risk (2FA / Review)": "#f59e0b",
                "High Risk (Blocked)": "#ef4444"
            }
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("Live Operational Status")
        is_ready = predictor.is_ready
        st.success(f"✅ Prediction Engine: **{'Online (Model Loaded)' if is_ready else 'Degraded (Train Model First)'}**")
        st.info(f"🛡️ Active Champion Model: **{metrics_data.get('selected_model', 'Random Forest') if metrics_data else 'Trained Model'}**")
        st.warning(f"⚡ Imbalance Handling: **Stratified Split + Cost-Sensitive Balanced Weights**")
        st.write("---")
        st.write("### Recent Quick Activity Feed")
        recent_txns = database.get_transactions(limit=5)
        if recent_txns:
            df_recent = pd.DataFrame(recent_txns)[["transaction_id", "timestamp", "amount", "fraud_probability", "risk_level"]]
            df_recent["amount"] = df_recent["amount"].apply(lambda x: f"€{x:,.2f}")
            df_recent["fraud_probability"] = df_recent["fraud_probability"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions logged yet. Test a transaction in the Real-Time Detection tab!")


# -------------------------------------------------------------
# TAB 2: REAL-TIME DETECTION
# -------------------------------------------------------------
elif navigation == "⚡ Real-Time Detection":
    st.title("⚡ Real-Time Transaction Fraud Prediction")
    st.markdown("Submit transaction details to evaluate fraud probability, risk category, and feature explanations.")

    if not predictor.is_ready:
        st.error("⚠️ Model artifacts not found! Please run `python train_model.py` to train the models.")
        st.stop()

    # Preset Sample Selector for quick testing
    preset_col1, preset_col2 = st.columns([2, 1])
    with preset_col1:
        st.subheader("1. Select a Verified Test Preset or Enter Custom Values")
    with preset_col2:
        preset_choice = "Custom Manual Input"
        if not sample_df.empty:
            options = ["Custom Manual Input"] + [
                f"Sample #{i+1} - {'[CONFIRMED FRAUD]' if row['Class']==1 else '[LEGITIMATE]'}: €{row['Amount']:.2f}"
                for i, row in sample_df.iterrows()
            ]
            selected_option = st.selectbox("Load Authentic Sample Preset", options)
            if selected_option != "Custom Manual Input":
                preset_idx = int(selected_option.split("#")[1].split(" ")[0]) - 1
                loaded_row = sample_df.iloc[preset_idx].to_dict()
            else:
                loaded_row = {}
        else:
            loaded_row = {}

    # Form with inputs
    with st.form("transaction_input_form"):
        st.markdown("##### Core Transaction Parameters")
        c1, c2 = st.columns(2)
        with c1:
            input_amount = st.number_input(
                "Transaction Amount (€)",
                min_value=0.01,
                max_value=50000.0,
                value=float(loaded_row.get("Amount", 149.62)),
                step=10.0,
                help="Actual financial value in Euros (€)"
            )
        with c2:
            input_time = st.number_input(
                "Time Offset (Seconds)",
                min_value=0.0,
                max_value=200000.0,
                value=float(loaded_row.get("Time", 406.0)),
                step=100.0,
                help="Seconds elapsed since the initial benchmark transaction"
            )

        with st.expander("🛠️ Advanced PCA Latent Components (V1 to V28)", expanded=False):
            st.caption("PCA features are orthogonal projections of cardholder and merchant features. Pre-filled with realistic distributions.")
            pca_inputs = {}
            # Display in 4 grid columns
            cols = st.columns(4)
            for i in range(1, 29):
                feat_name = f"V{i}"
                default_val = float(loaded_row.get(feat_name, 0.0))
                with cols[(i - 1) % 4]:
                    pca_inputs[feat_name] = st.number_input(
                        feat_name,
                        value=default_val,
                        step=0.1,
                        format="%.3f"
                    )

        submit_btn = st.form_submit_button("🛡️ Run Fraud Detection Analysis", use_container_width=True)

    if submit_btn:
        payload = {"Time": input_time, "Amount": input_amount, **pca_inputs}
        
        with st.spinner("Analyzing transaction patterns with Machine Learning..."):
            result = predictor.predict_single(payload, save_to_db=True, source="dashboard_realtime")

        st.write("---")
        st.subheader("Analysis & Risk Assessment Results")

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])

        prob = result["fraud_probability"]
        risk = result["risk_level"]

        with res_col1:
            st.metric(label="Estimated Fraud Probability", value=f"{prob * 100:.2f}%")
            if risk == "HIGH":
                st.markdown("<div class='badge-high'>🚨 HIGH RISK TRANSACTION</div>", unsafe_allow_html=True)
            elif risk == "MEDIUM":
                st.markdown("<div class='badge-medium'>⚠️ MEDIUM RISK - VERIFICATION NEEDED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-low'>✅ LOW RISK - CLEARED</div>", unsafe_allow_html=True)

        with res_col2:
            st.metric(label="System Decision", value=result["prediction_label"])
            st.caption(f"Transaction ID: `{result['transaction_id']}`")
            st.caption(f"Database Record: **Saved & Indexed**")

        with res_col3:
            # Plotly Gauge Chart for visual risk score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%"},
                title={'text': "Fraud Risk Gauge", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if risk == "HIGH" else ("#f59e0b" if risk == "MEDIUM" else "#10b981")},
                    'steps': [
                        {'range': [0, predictor.threshold_low * 100], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [predictor.threshold_low * 100, predictor.threshold_high * 100], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [predictor.threshold_high * 100, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Explainability Section
        st.write("---")
        st.subheader("🧠 Explainable AI: Why Was This Decision Reached?")
        expl = result.get("explanation", {})
        st.markdown(f"**Executive Synthesis:** {expl.get('summary', '')}")

        top_feats = expl.get("top_features", [])
        if top_feats:
            df_impact = pd.DataFrame(top_feats)
            fig_bar = px.bar(
                df_impact,
                x="impact_score",
                y="feature",
                orientation="h",
                color="impact_score",
                color_continuous_scale="Reds" if risk != "LOW" else "Greens",
                title="Top 5 Features Influencing the Fraud Assessment",
                labels={"impact_score": "Relative Anomaly Impact Score", "feature": "Feature"}
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(t=30, b=20, l=40, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(f"ℹ️ {expl.get('disclaimer', '')}")


# -------------------------------------------------------------
# TAB 3: BULK CSV ANALYSIS
# -------------------------------------------------------------
elif navigation == "📁 Bulk CSV Analysis":
    st.title("📁 Bulk CSV Transaction Batch Processing")
    st.markdown("Upload bulk transaction files for automated high-speed fraud screening and reporting.")

    if not predictor.is_ready:
        st.error("⚠️ Model artifacts not found! Run `python train_model.py` first.")
        st.stop()

    uploaded_file = st.file_uploader("Upload CSV File (must contain Amount and V1-V28 columns)", type=["csv"])

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        # Download Sample Template Button
        if not sample_df.empty:
            template_csv = sample_df.drop(columns=["Class"], errors="ignore").to_csv(index=False)
            st.download_button(
                "📥 Download CSV Template",
                data=template_csv,
                file_name="fraud_detection_template.csv",
                mime="text/csv"
            )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file successfully parsed: **{len(input_df):,} rows**, **{len(input_df.columns)} columns**.")

            with st.spinner("Processing batch transactions through ML pipeline..."):
                t0 = time.time()
                scored_df, batch_stats = predictor.predict_batch(input_df, save_to_db=False)
                proc_time = time.time() - t0

            st.write("---")
            st.subheader("Batch Analytics Summary")
            
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Scanned", f"{batch_stats['total_processed']:,}")
            b2.metric("Flagged Suspicious", f"{batch_stats['flagged_count']:,}", delta=f"{batch_stats['fraud_rate_percentage']:.2f}% rate")
            b3.metric("High-Risk Blocked", f"{batch_stats['high_risk_count']:,}")
            b4.metric("Flagged Volume", f"€{batch_stats['flagged_amount_total']:,.2f}")

            st.caption(f"Processed in **{proc_time:.3f} seconds** ({batch_stats['total_processed'] / max(proc_time, 0.001):,.0f} transactions/sec).")

            # Result Data Table
            st.write("### Scored Transaction Table")
            display_cols = ["Time", "Amount", "fraud_probability", "risk_level", "prediction", "is_flagged"] + [f"V{i}" for i in range(1, 6)]
            st.dataframe(scored_df[display_cols], use_container_width=True)

            # Export button
            csv_output = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Predictions CSV",
                data=csv_output,
                file_name=f"scored_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Option to persist into SQLite
            if st.button("💾 Persist Batch Records to SQLite Audit History"):
                with st.spinner("Saving records to database..."):
                    predictor.predict_batch(input_df, save_to_db=True, source="batch_csv_upload")
                st.success("Successfully saved all batch transactions to SQLite audit history!")

        except Exception as e:
            st.error(f"Error during bulk CSV processing: {str(e)}")


# -------------------------------------------------------------
# TAB 4: FRAUD ANALYTICS
# -------------------------------------------------------------
elif navigation == "📈 Fraud Analytics":
    st.title("📈 Fraud Analytics & Pattern Exploration")
    st.markdown("Visual analysis of transaction patterns, distributions, and anomaly correlations.")

    raw_data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "creditcard.csv")
    if not os.path.exists(raw_data_file):
        st.warning("Dataset file not found at data/creditcard.csv. Run `python scripts/download_or_generate_data.py` to prepare data.")
        st.stop()

    @st.cache_data
    def load_analytics_data():
        df = pd.read_csv(raw_data_file)
        # Sample for fast interactive visualization if dataset is large
        if len(df) > 20000:
            df_fraud = df[df["Class"] == 1]
            df_legit = df[df["Class"] == 0].sample(20000, random_state=42)
            return pd.concat([df_legit, df_fraud])
        return df

    data_sample = load_analytics_data()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Amount Distribution (Log-Scaled)")
        fig_hist = px.histogram(
            data_sample,
            x="Amount",
            color="Class",
            nbins=50,
            log_y=True,
            barmode="overlay",
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            labels={"Class": "Transaction Type", "Amount": "Amount (€)"}
        )
        fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Key Discriminative Feature: V14 vs V17")
        fig_scatter = px.scatter(
            data_sample,
            x="V14",
            y="V17",
            color="Class",
            opacity=0.6,
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            labels={"Class": "Class (0=Legit, 1=Fraud)"}
        )
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Transaction Activity Over Time (48-Hour Cycle)")
    data_sample["Hour"] = (data_sample["Time"] / 3600) % 24
    fig_time = px.histogram(
        data_sample,
        x="Hour",
        color="Class",
        barmode="group",
        nbins=24,
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        labels={"Hour": "Hour of the Day", "Class": "Class"}
    )
    fig_time.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_time, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: MODEL PERFORMANCE
# -------------------------------------------------------------
elif navigation == "🧠 Model Performance":
    st.title("🧠 Model Performance & Comparative Evaluation")
    st.markdown("Exhaustive metrics on imbalanced classification, comparing Baseline vs Champion models.")

    if not metrics_data:
        st.warning("Model metrics not found. Run `python train_model.py` to train and evaluate models.")
        st.stop()

    models = metrics_data.get("models", {})

    st.subheader("Comparative Evaluation Leaderboard")
    leaderboard = []
    for name, m in models.items():
        cm = m["confusion_matrix"]
        leaderboard.append({
            "Model Name": name,
            "PR-AUC (Avg Precision)": m["pr_auc"],
            "F1-Score": m["f1_score"],
            "Recall (Fraud Caught)": m["recall"],
            "Precision (Accuracy of Flags)": m["precision"],
            "ROC-AUC": m["roc_auc"],
            "Training Time (s)": m["training_time_seconds"],
            "True Positives": cm["true_positives"],
            "False Positives": cm["false_positives"],
            "False Negatives": cm["false_negatives"]
        })

    df_lb = pd.DataFrame(leaderboard).sort_values("PR-AUC (Avg Precision)", ascending=False)
    st.dataframe(df_lb, use_container_width=True, hide_index=True)

    st.write("---")
    c_curve1, c_curve2 = st.columns(2)

    with c_curve1:
        st.subheader("Precision-Recall (PR) Curves")
        fig_pr = go.Figure()
        for name, m in models.items():
            pr_data = m.get("pr_curve", {})
            if pr_data:
                fig_pr.add_trace(go.Scatter(
                    x=pr_data["recall"],
                    y=pr_data["precision"],
                    mode="lines",
                    name=f"{name} (PR-AUC: {m['pr_auc']})"
                ))
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with c_curve2:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        for name, m in models.items():
            roc_data = m.get("roc_curve", {})
            if roc_data:
                fig_roc.add_trace(go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    name=f"{name} (ROC-AUC: {m['roc_auc']})"
                ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random Guess"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Explanation of trade-offs and limitations
    with st.expander("📚 Data Science Discussion: Accuracy vs. PR-AUC & Business Trade-offs", expanded=True):
        st.markdown("""
        **Why Accuracy Fails for Fraud Detection:**
        In a dataset with 99.83% legitimate transactions, a trivial rule stating *"always predict legitimate"* yields **99.83% accuracy**, yet allows **100% of frauds to succeed unnoticed**.
        
        **The Financial Trade-off:**
        - **Prioritizing Recall (Catching all fraud)**: Reduces financial fraud chargebacks, but may increase False Positives ($FP$), causing legitimate users to experience friction or declined cards.
        - **Prioritizing Precision (Fewer false alarms)**: Prevents customer friction, but may increase False Negatives ($FN$), allowing actual fraudulent transactions through.
        - **Precision-Recall AUC (PR-AUC)** is the true benchmark of ranking quality across imbalanced positive classes.
        """)


# -------------------------------------------------------------
# TAB 6: TRANSACTION AUDIT HISTORY
# -------------------------------------------------------------
elif navigation == "📜 Transaction Audit History":
    st.title("📜 Transaction Audit History & Inspection")
    st.markdown("Searchable SQLite transaction log recording every evaluated transaction, risk level, and explanation.")

    c_f1, c_f2, c_f3 = st.columns([2, 1, 1])
    with c_f1:
        search_term = st.text_input("🔍 Search by Transaction ID or Source", "")
    with c_f2:
        risk_filter = st.selectbox("Filter Risk Level", ["ALL", "HIGH", "MEDIUM", "LOW"])
    with c_f3:
        flagged_filter = st.selectbox("Flag Status", ["ALL", "Flagged Only", "Unflagged Only"])

    rf_arg = None if risk_filter == "ALL" else risk_filter
    ff_arg = None if flagged_filter == "ALL" else (True if flagged_filter == "Flagged Only" else False)

    txns = database.get_transactions(
        limit=200,
        risk_level=rf_arg,
        is_flagged=ff_arg,
        search_query=search_term if search_term else None
    )

    if txns:
        df_hist = pd.DataFrame(txns)
        display_hist = df_hist[[
            "transaction_id", "timestamp", "amount", "fraud_probability",
            "risk_level", "prediction", "is_flagged", "source"
        ]].copy()
        
        display_hist["amount"] = display_hist["amount"].apply(lambda x: f"€{x:,.2f}")
        display_hist["fraud_probability"] = display_hist["fraud_probability"].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(display_hist, use_container_width=True, hide_index=True)

        st.caption(f"Showing **{len(txns)}** records.")
        
        # Download audit log button
        csv_hist = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Complete Audit Log (CSV)",
            data=csv_hist,
            file_name="transaction_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching transaction records found in SQLite database.")
