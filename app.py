import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from model.train_model import load_model
from model.similarity import get_similar_tickets
from utils.confidence import compute_confidence, get_confidence_label
from utils.decision import decide, get_priority

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDesk AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}

/* HEADER */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    color: #f8fafc;
    font-size: 1.6rem;
    margin: 0;
    font-weight: 600;
}
.main-header p {
    color: #94a3b8;
    margin-top: 4px;
    font-size: 0.9rem;
}

/* METRIC CARDS */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    transition: all 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}
.metric-card .val {
    font-size: 1.8rem;
    font-weight: 600;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #64748b;
}

/* TICKET CARDS */
.ticket-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    transition: 0.2s;
}
.ticket-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* STATUS BADGES */
.status-auto {
    background:#dcfce7;
    color:#166534;
    padding:4px 10px;
    border-radius:20px;
    font-size:0.75rem;
    font-weight:600;
}
.status-pending {
    background:#fef9c3;
    color:#854d0e;
    padding:4px 10px;
    border-radius:20px;
    font-size:0.75rem;
    font-weight:600;
}
.status-critical {
    background:#fee2e2;
    color:#991b1b;
    padding:4px 10px;
    border-radius:20px;
    font-size:0.75rem;
    font-weight:600;
}

/* CONFIDENCE BAR */
.conf-bar-wrap {
    background:#e2e8f0;
    border-radius:6px;
    height:6px;
}
.conf-bar-fill {
    height:6px;
    border-radius:6px;
}

/* BUTTONS */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    border: none;
    height: 42px;
}

/* PRIMARY BUTTON (FORM - ANALYZE) */
div[data-testid="stForm"] button {
    background: #e2e8f0 !important;   /* soft blue-grey */
    color: #1e3a5f !important;        /* dark blue text */
    border: 1px solid #cbd5e1 !important;
    font-weight: 600;
}

div[data-testid="stForm"] button:hover {
    background: #cbd5e1 !important;   /* slightly darker on hover */
    color: #0f172a !important;
}

/* SECONDARY BUTTONS */
.stButton>button[kind="secondary"] {
    background: #f1f5f9;
    color: #1e293b;
}

/* INPUT FIELDS */
textarea, input {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* SIDEBAR (LIGHT — IMPORTANT CHANGE) */
div[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
div[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* SIDEBAR NAV HIGHLIGHT */
div[data-testid="stSidebar"] .stRadio > div {
    gap: 6px;
}
div[data-testid="stSidebar"] label {
    padding: 6px 10px;
    border-radius: 6px;
}
div[data-testid="stSidebar"] label:hover {
    background: #f1f5f9;
}

/* DIVIDERS */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1rem 0;
}

/* REMOVE EXTRA PADDING */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── session state init ────────────────────────────────────────────────────────
if 'tickets' not in st.session_state:
    st.session_state.tickets = []
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'classifier' not in st.session_state:
    with st.spinner("Loading AI models…"):
        st.session_state.classifier = load_model()

clf = st.session_state.classifier

# ── helpers ───────────────────────────────────────────────────────────────────

def analyze_ticket(text: str) -> dict:
    # Get classifier prediction
    predicted_category = clf.predict([text])[0]
    proba = clf.predict_proba([text])[0].max()

    # Get similarity results
    similar, sim_score = get_similar_tickets(text, top_k=3)

    # HYBRID CATEGORY LOGIC
    if similar and sim_score > 0.3:
        category = similar[0]['category']
    else:
        category = predicted_category

    confidence = compute_confidence(sim_score, proba)
    decision = decide(confidence, text)
    priority = get_priority(category)
    resolution = similar[0]['resolution'] if similar else "No resolution found."

    ticket_id = f"TKT-{str(uuid.uuid4())[:6].upper()}"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    ticket = {
        "id": ticket_id,
        "text": text,
        "category": category,
        "priority": priority,
        "confidence": confidence,
        "classification_prob": round(proba, 4),
        "similarity_score": round(sim_score, 4),
        "status": decision["status"],
        "auto": decision["auto"],
        "is_critical": decision["is_critical"],
        "reason": decision["reason"],
        "resolution": resolution,
        "similar_tickets": similar,
        "timestamp": ts,
        "human_override": None,
    }
    return ticket


def conf_bar_html(conf: float) -> str:
    pct = int(conf * 100)
    color = "#22c55e" if conf >= 0.8 else "#f59e0b" if conf >= 0.5 else "#ef4444"
    return f"""
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    <small style="color:#64748b;font-family:'IBM Plex Mono',monospace">{pct}%</small>
    """


def status_badge(status: str, is_critical: bool = False) -> str:
    if is_critical:
        return '<span class="status-critical">🔴 Critical</span>'
    if "Auto" in status:
        return '<span class="status-auto">Auto-Resolved</span>'
    return '<span class="status-pending">Pending Review</span>'


# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", width=200)
    st.markdown("---")
    role = st.selectbox("Select Role", ["User", "Support Agent"])
    st.caption(f"Logged in as: {role}")
    st.markdown("---")
    if role == "User":
        page = st.radio(
            "Navigation",
            ["New Ticket", "My Tickets"],
            label_visibility="collapsed"
        )
    else:
        page = st.radio(
            "Navigation",
            ["Dashboard", "Tickets", "Pending Reviews", "Analytics", "Audit Log"],
            label_visibility="collapsed"
        )
    st.markdown("---")
    if role == "Support Agent":
        total = len(st.session_state.tickets)
        auto  = sum(1 for t in st.session_state.tickets if t['auto'])
        pend  = sum(1 for t in st.session_state.tickets if not t['auto'])
        st.markdown(f"**Total:** {total} tickets")
        st.markdown(f"**Auto-Resolved:** {auto}")
        st.markdown(f"**Pending:** {pend}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="main-header">
      <h1>SmartDesk AI — Dashboard</h1>
      <p>Confidence-Based Intelligent Ticket Automation with Human-in-the-Loop</p>
    </div>
    """, unsafe_allow_html=True)

    tickets = st.session_state.tickets
    total   = len(tickets)
    auto    = sum(1 for t in tickets if t['auto'])
    pend    = sum(1 for t in tickets if not t['auto'] and not t['is_critical'])
    crit    = sum(1 for t in tickets if t['is_critical'])
    avg_conf = round(np.mean([t['confidence'] for t in tickets]) * 100, 1) if tickets else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, total, "Total Tickets"),
        (c2, auto, "Auto-Resolved"),
        (c3, pend, "Pending Review"),
        (c4, crit, "Critical"),
        (c5, f"{avg_conf}%", "Avg Confidence"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if tickets:
        st.subheader("Recent Tickets")
        for t in reversed(tickets[-5:]):
            col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1.5])
            col1.write(f"**{t['id']}** — {t['text'][:60]}…")
            col2.write(t['category'])
            col3.markdown(conf_bar_html(t['confidence']), unsafe_allow_html=True)
            col4.markdown(status_badge(t['status'], t['is_critical']), unsafe_allow_html=True)
    else:
        st.info("No tickets yet. Use **New Ticket** to submit one.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEW TICKET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "New Ticket":
    st.markdown("""
    <div class="main-header">
      <h1>Submit New Ticket</h1>
      <p>Describe your IT issue and let the AI analyze it instantly</p>
    </div>""", unsafe_allow_html=True)

    with st.form("ticket_form"):
        ticket_text = st.text_area(
            "Describe your issue",
            placeholder="e.g. VPN not connecting after Windows update…",
            height=130
        )
        submitted = st.form_submit_button("Analyze Ticket", use_container_width=True)

    if submitted and ticket_text.strip():
        with st.spinner("Analyzing with AI…"):
            ticket = analyze_ticket(ticket_text.strip())
            st.session_state.tickets.append(ticket)

            audit_entry = {
                "ticket_id": ticket['id'],
                "ai_decision": ticket['status'],
                "confidence": ticket['confidence'],
                "human_override": "No",
                "timestamp": ticket['timestamp'],
            }
            st.session_state.audit_log.append(audit_entry)

        conf_info = get_confidence_label(ticket['confidence'])

        st.markdown("---")
        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Ticket ID:** `{ticket['id']}`")
            st.markdown(f"**Category:** {ticket['category']}")
            st.markdown(f"**Priority:** `{ticket['priority']}`")
            st.markdown(f"**Timestamp:** {ticket['timestamp']}")
        with col2:
            st.markdown(f"**Confidence:** {conf_info['emoji']} {ticket['confidence']:.0%} ({conf_info['label']})")
            st.markdown(f"**Similarity Score:** {ticket['similarity_score']:.0%}")
            st.markdown(f"**Classification Prob:** {ticket['classification_prob']:.0%}")
            st.markdown(status_badge(ticket['status'], ticket['is_critical']), unsafe_allow_html=True)

        st.markdown("---")

        # Decision result
        if ticket['auto']:
            st.success(f"**Auto-Resolved** — {ticket['reason']}")
            st.markdown(f"**Suggested Resolution:**\n\n> {ticket['resolution']}")
        else:
            st.warning(f"**{ticket['status']}** — {ticket['reason']}")
            st.markdown(f"**Suggested Resolution:**\n\n> {ticket['resolution']}")
        st.info("This ticket has been forwarded to the support team for review.")

        # Explainability
        st.markdown("---")
        st.subheader("Explainability — Top Similar Tickets")
        for i, sim in enumerate(ticket['similar_tickets'], 1):
            pct = int(sim['similarity'] * 100)
            color = "#22c55e" if sim['similarity'] >= 0.8 else "#f59e0b" if sim['similarity'] >= 0.5 else "#94a3b8"
            st.markdown(f"""
            <div class="ticket-card">
              <h4>#{i} — {sim['ticket_text']}</h4>
              <p><b>Category:</b> {sim['category']} &nbsp;|&nbsp;
                 <b>Similarity:</b> <span style="color:{color};font-weight:700">{pct}%</span></p>
              <p><b>Resolution:</b> {sim['resolution']}</p>
            </div>""", unsafe_allow_html=True)
        st.toast("Ticket submitted successfully!")

    elif submitted:
        st.warning("Please enter a ticket description.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TICKETS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Tickets" or page == "My Tickets":
    st.info("Track your submitted tickets and their current status here.")
    if role == "User":
        title = "My Tickets"
        subtitle = "Track your submitted tickets and their status"
    else:
        title = "All Tickets"
        subtitle = "Full ticket registry with AI classification results"

    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

    tickets = st.session_state.tickets
    if not tickets:
        st.info("No tickets yet. Submit one via **New Ticket**.")
    else:
        # Filter
        cats = ["All"] + list({t['category'] for t in tickets})
        col1, col2 = st.columns([2, 4])
        cat_filter = col1.selectbox("Filter by Category", cats)

        filtered = tickets if cat_filter == "All" else [t for t in tickets if t['category'] == cat_filter]

        st.markdown(f"Showing **{len(filtered)}** tickets")
        st.markdown("---")

        for t in reversed(filtered):
            c1, c2, c3, c4, c5 = st.columns([1.2, 3, 1.5, 2, 2])
            c1.markdown(f"`{t['id']}`")
            c2.write(t['text'][:55] + ("…" if len(t['text']) > 55 else ""))
            if role != "User":
                c3.write(t['category'])
            c4.markdown(conf_bar_html(t['confidence']), unsafe_allow_html=True)
            c5.markdown(f"<small style='color:#64748b'>Status: {t['status']}</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PENDING REVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pending Reviews":
    st.markdown("""
    <div class="main-header">
      <h1>Pending Review Queue</h1>
      <p>Low-confidence or critical tickets requiring human attention</p>
    </div>""", unsafe_allow_html=True)

    pending = [t for t in st.session_state.tickets if not t['auto'] and t['status'] not in ("Escalated", "Auto-Resolved")]

    if not pending:
        st.success("No tickets pending review!")
    else:
        st.warning(f"**{len(pending)} ticket(s)** need your attention.")
        for i, t in enumerate(reversed(pending)):
            with st.expander(f"{t['id']} — {t['text'][:70]}…  |  Confidence: {t['confidence']:.0%}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Category:** {t['category']}")
                    st.markdown(f"**Priority:** `{t['priority']}`")
                    st.markdown(f"**Submitted:** {t['timestamp']}")
                with col2:
                    conf_info = get_confidence_label(t['confidence'])
                    st.markdown(f"**Confidence:** {conf_info['emoji']} {t['confidence']:.0%}")
                    st.markdown(f"**Reason:** {t['reason']}")

                st.markdown(f"**AI Suggested Resolution:**\n> {t['resolution']}")

                col_a, col_b = st.columns(2)
                idx = st.session_state.tickets.index(t)
                if col_a.button("Approve", key=f"apr_{t['id']}", use_container_width=True, type="primary"):
                    st.session_state.tickets[idx]['status'] = "Auto-Resolved"
                    st.session_state.tickets[idx]['auto'] = True
                    st.session_state.tickets[idx]['human_override'] = "Approved"

                    for entry in st.session_state.audit_log:
                        if entry['ticket_id'] == t['id']:
                            entry['human_override'] = "Yes"
                            entry['ai_decision'] = "Auto-Resolved"

                    st.toast(f"Ticket {t['id']} approved & user notified")
                    st.rerun()

                if col_b.button("Reject", key=f"rej_{t['id']}", use_container_width=True):
                    st.session_state.tickets[idx]['status'] = "Escalated"
                    st.session_state.tickets[idx]['human_override'] = "Rejected"

                    for entry in st.session_state.audit_log:
                        if entry['ticket_id'] == t['id']:
                            entry['human_override'] = "Yes"
                            entry['ai_decision'] = "Escalated"

                    st.toast(f"Ticket {t['id']} escalated & user notified")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("""
    <div class="main-header">
      <h1>Analytics</h1>
      <p>AI performance metrics and ticket insights</p>
    </div>""", unsafe_allow_html=True)

    tickets = st.session_state.tickets

    if not tickets:
        st.info("Submit some tickets to see analytics.")
    else:
        total = len(tickets)
        auto  = sum(1 for t in tickets if t['auto'])
        overrides = sum(1 for t in tickets if t.get('human_override') in ("Approved", "Rejected"))
        avg_conf = np.mean([t['confidence'] for t in tickets])

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            (f"{auto/total*100:.1f}%", "AI Auto-Resolve Rate"),
            (f"{overrides/total*100:.1f}%", "Human Override Rate"),
            (f"{avg_conf*100:.1f}%", "Avg Confidence"),
            (f"{total}", "Tickets Analyzed"),
        ]
        for col, (val, lbl) in zip([col1, col2, col3, col4], metrics):
            col.markdown(f"""<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_l, col_r = st.columns(2)

        # Category breakdown
        with col_l:
            st.subheader("Tickets by Category")
            cats = {}
            for t in tickets:
                cats[t['category']] = cats.get(t['category'], 0) + 1
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.barh(list(cats.keys()), list(cats.values()), color="#3b82f6")
            ax.set_xlabel("Count")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Confidence distribution
        with col_r:
            st.subheader("Confidence Score Distribution")
            confs = [t['confidence'] for t in tickets]
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.hist(confs, bins=10, color="#22c55e", edgecolor="white", range=(0, 1))
            ax2.axvline(0.8, color="#ef4444", linestyle="--", label="Threshold (0.8)")
            ax2.set_xlabel("Confidence Score")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # Status pie
        st.markdown("---")
        st.subheader("Resolution Status Breakdown")
        statuses = {}
        for t in tickets:
            statuses[t['status']] = statuses.get(t['status'], 0) + 1

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        colors = {"Auto-Resolved": "#22c55e", "Needs Human Review": "#f59e0b", "Escalated": "#ef4444"}
        pie_colors = [colors.get(s, "#94a3b8") for s in statuses.keys()]
        ax3.pie(statuses.values(), labels=statuses.keys(), colors=pie_colors, autopct='%1.0f%%', startangle=90)
        ax3.axis('equal')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    st.markdown("""
    <div class="main-header">
      <h1>Audit Log</h1>
      <p>Complete traceability of every AI decision and human override</p>
    </div>""", unsafe_allow_html=True)

    logs = st.session_state.audit_log
    if not logs:
        st.info("No audit entries yet.")
    else:
        df_log = pd.DataFrame(logs)
        df_log.columns = ["Ticket ID", "AI Decision", "Confidence", "Human Override", "Timestamp"]
        df_log["Confidence"] = df_log["Confidence"].apply(lambda x: f"{x:.0%}")
        st.dataframe(df_log[::-1].reset_index(drop=True), use_container_width=True)

        csv = df_log.to_csv(index=False)
        st.download_button("Download Audit Log CSV", csv, "audit_log.csv", "text/csv")
