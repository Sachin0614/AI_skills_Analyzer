import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="AI Resume Analyzer 3D",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(124,58,237,0.35), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(14,165,233,0.28), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(16,185,129,0.18), transparent 35%),
        linear-gradient(135deg, #050510 0%, #09091f 45%, #020617 100%);
    color: white;
    overflow-x: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 45px 45px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.9), transparent);
    pointer-events: none;
}

.hero {
    position: relative;
    padding: 55px 35px;
    margin-bottom: 35px;
    border-radius: 34px;
    text-align: center;
    background:
        linear-gradient(145deg, rgba(99,102,241,0.95), rgba(124,58,237,0.9), rgba(236,72,153,0.75));
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow:
        0 35px 100px rgba(124,58,237,0.45),
        inset 0 1px 0 rgba(255,255,255,0.35);
    transform-style: preserve-3d;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    top: -100px;
    right: -70px;
    background: rgba(255,255,255,0.18);
    border-radius: 50%;
    filter: blur(5px);
}

.hero::after {
    content: "🚀";
    position: absolute;
    font-size: 90px;
    right: 65px;
    top: 45px;
    opacity: 0.20;
    transform: rotate(-25deg);
    animation: floatRocket 4s ease-in-out infinite;
}

@keyframes floatRocket {
    0%, 100% { transform: translateY(0) rotate(-25deg); }
    50% { transform: translateY(-18px) rotate(-15deg); }
}

.hero h1 {
    font-size: 3.4rem;
    font-weight: 900;
    color: white;
    margin: 0;
    letter-spacing: -1.5px;
    text-shadow: 0 10px 35px rgba(0,0,0,0.35);
}

.hero p {
    color: rgba(255,255,255,0.88);
    font-size: 1.15rem;
    margin-top: 13px;
}

.glass-card, .stat-card, .res-header {
    background: linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.035));
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow:
        0 22px 60px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.18);
    backdrop-filter: blur(18px);
    border-radius: 24px;
}

.stat-card {
    padding: 24px 18px;
    text-align: center;
    transition: all 0.28s ease;
    transform: perspective(900px) rotateX(0deg) rotateY(0deg);
}

.stat-card:hover {
    transform: perspective(900px) rotateX(6deg) rotateY(-6deg) translateY(-8px);
    box-shadow:
        0 32px 90px rgba(99,102,241,0.35),
        inset 0 1px 0 rgba(255,255,255,0.22);
}

.stat-num {
    font-size: 2.45rem;
    font-weight: 900;
    background: linear-gradient(135deg, #67e8f9, #a78bfa, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    color: rgba(255,255,255,0.58);
    font-size: 0.77rem;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sec-title {
    color: white;
    font-size: 1.32rem;
    font-weight: 850;
    margin: 30px 0 16px;
    padding-left: 14px;
    border-left: 5px solid #8b5cf6;
    text-shadow: 0 5px 20px rgba(139,92,246,0.35);
}

.winner {
    position: relative;
    background:
        linear-gradient(135deg, rgba(245,158,11,0.95), rgba(249,115,22,0.93), rgba(236,72,153,0.85));
    border-radius: 28px;
    padding: 28px 30px;
    text-align: center;
    margin: 30px 0;
    box-shadow:
        0 28px 85px rgba(245,158,11,0.35),
        inset 0 1px 0 rgba(255,255,255,0.30);
    border: 1px solid rgba(255,255,255,0.22);
    overflow: hidden;
}

.winner h2 {
    margin: 0;
    font-weight: 900;
    color: white;
    font-size: 1.8rem;
}

.winner p {
    margin-top: 8px;
    color: rgba(255,255,255,0.9);
}

.skill-match, .skill-match-keyword, .skill-miss {
    border-radius: 15px;
    padding: 12px 16px;
    margin: 7px 0;
    font-size: 0.9rem;
    transition: all 0.2s ease;
    box-shadow: 0 10px 26px rgba(0,0,0,0.25);
}

.skill-match:hover, .skill-match-keyword:hover, .skill-miss:hover {
    transform: translateY(-3px) scale(1.01);
}

.skill-match {
    background: linear-gradient(135deg, rgba(16,185,129,0.20), rgba(16,185,129,0.05));
    border: 1px solid rgba(52,211,153,0.45);
    color: #6ee7b7;
}

.skill-match-keyword {
    background: linear-gradient(135deg, rgba(34,197,94,0.33), rgba(14,165,233,0.10));
    border: 1px solid rgba(110,231,183,0.70);
    color: #bbf7d0;
    font-weight: 800;
}

.skill-miss {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(236,72,153,0.07));
    border: 1px solid rgba(248,113,113,0.45);
    color: #fca5a5;
}

section[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035)) !important;
    border: 2px dashed rgba(168,85,247,0.65) !important;
    border-radius: 24px !important;
    padding: 28px !important;
    box-shadow: 0 18px 55px rgba(0,0,0,0.35) !important;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #06b6d4, #8b5cf6, #ec4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 14px 26px !important;
    font-weight: 850 !important;
    width: 100% !important;
    box-shadow: 0 16px 45px rgba(139,92,246,0.45) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-4px) scale(1.015) !important;
    box-shadow: 0 25px 70px rgba(236,72,153,0.45) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(30,27,75,0.94)) !important;
    border-right: 1px solid rgba(255,255,255,0.11);
}

[data-testid="stExpander"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.085), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}

h1, h2, h3, h4, p, li, label, span {
    color: white;
}

.stDataFrame {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🔄 Loading AI Model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text.strip()


def analyze_resume(model, skills, resume_text, threshold):
    matched, missing = [], []
    resume_lower = resume_text.lower()

    sentences = [
        s.strip()
        for s in resume_text.replace("\n", ". ").split(".")
        if len(s.strip()) > 8
    ]

    if not sentences:
        return matched, missing

    res_embeddings = model.encode(sentences, convert_to_tensor=True)

    for skill in skills:
        skill_lower = skill.lower()
        keyword_found = skill_lower in resume_lower

        skill_emb = model.encode(skill, convert_to_tensor=True)
        scores = util.cos_sim(skill_emb, res_embeddings)[0]
        best_score = float(scores.max())
        best_sentence = sentences[int(scores.argmax())]

        if keyword_found:
            matched.append({
                "skill": skill,
                "score": max(round(best_score * 100), 78),
                "reason": best_sentence[:130],
                "type": "keyword"
            })
        elif best_score >= threshold:
            matched.append({
                "skill": skill,
                "score": round(best_score * 100),
                "reason": best_sentence[:130],
                "type": "semantic"
            })
        else:
            missing.append({
                "skill": skill,
                "score": round(best_score * 100)
            })

    return matched, missing


def score_color(pct):
    if pct >= 70:
        return "#22c55e"
    elif pct >= 45:
        return "#f59e0b"
    return "#ef4444"


with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("AI matching settings")

    threshold = st.slider(
        "Semantic Match Sensitivity",
        0.30, 0.65, 0.50, 0.05,
        help="Higher = strict matching | Lower = loose matching"
    )

    st.markdown("---")
    st.markdown("### 🎯 Required Skills")
    jd_input = st.text_area(
        "Enter one skill per line",
        value="""Python
Machine Learning
Deep Learning
Natural Language Processing
LangChain
FastAPI
Docker
AWS
Git
SQL
Data Analysis
Computer Vision
TensorFlow
REST APIs
Prompt Engineering""",
        height=330
    )

    skills = [s.strip() for s in jd_input.strip().split("\n") if s.strip()]
    st.success(f"✅ {len(skills)} skills loaded")

    st.markdown("---")
    st.markdown("### Match Legend")
    st.markdown('<div class="skill-match-keyword">🔑 Keyword Match</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-match">🧠 Semantic Match</div>', unsafe_allow_html=True)
    st.markdown('<div class="skill-miss">❌ Missing Skill</div>', unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🚀 AI Resume Analyzer Pro</h1>
    <p>3D dashboard • AI skill matching • candidate ranking • visual reports</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="sec-title">📤 Upload Candidate Resumes</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload PDF resumes",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded:
    st.success(f"✅ {len(uploaded)} resume(s) uploaded successfully")

go_btn = st.button("🚀 Start AI Analysis")

if go_btn:
    if not uploaded:
        st.error("❌ Please upload at least one resume.")
        st.stop()

    if len(skills) < 2:
        st.error("❌ Please add skills in the sidebar.")
        st.stop()

    model = load_model()
    results = []

    progress = st.progress(0, "Starting analysis...")

    for i, file in enumerate(uploaded):
        progress.progress((i + 1) / len(uploaded), f"🔍 Reading {file.name}")

        text = extract_text(file)

        if text:
            matched, missing = analyze_resume(model, skills, text, threshold)
            pct = round(len(matched) / len(skills) * 100)

            results.append({
                "name": file.name.replace(".pdf", ""),
                "short": file.name.replace(".pdf", "")[:22],
                "matched": matched,
                "missing": missing,
                "score": len(matched),
                "total": len(skills),
                "pct": pct
            })

    progress.empty()

    if not results:
        st.error("❌ Text extract nahi ho paya. Scanned/image PDF ho sakti hai.")
        st.stop()

    results_sorted = sorted(results, key=lambda x: x["pct"], reverse=True)
    best = results_sorted[0]

    st.markdown('<p class="sec-title">📊 3D Overview</p>', unsafe_allow_html=True)

    cols = st.columns(min(len(results_sorted) + 1, 5))

    with cols[0]:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-num">{len(results_sorted)}</div>
                <div class="stat-label">Resumes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    for idx, r in enumerate(results_sorted[:4]):
        with cols[idx + 1]:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-num">{r['pct']}%</div>
                    <div class="stat-label">{r['short']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(f"""
    <div class="winner">
        <h2>🏆 Best Candidate: {best['name']}</h2>
        <p>{best['score']}/{best['total']} skills matched • {best['pct']}% overall match</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec-title">📈 Candidate Ranking Chart</p>', unsafe_allow_html=True)

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=[r["short"] for r in results_sorted],
        y=[r["pct"] for r in results_sorted],
        text=[f"{r['pct']}%" for r in results_sorted],
        textposition="outside",
        marker=dict(
            color=[score_color(r["pct"]) for r in results_sorted],
            line=dict(color="rgba(255,255,255,0.55)", width=2)
        ),
        hovertemplate="<b>%{x}</b><br>Match: %{y}%<extra></extra>"
    ))

    fig_bar.update_layout(
        height=430,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        yaxis=dict(
            range=[0, 110],
            title="Match %",
            gridcolor="rgba(255,255,255,0.10)"
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=30, b=20, l=30, r=30),
        showlegend=False
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    if len(results_sorted) > 1:
        st.markdown('<p class="sec-title">🕸️ AI Skill Radar</p>', unsafe_allow_html=True)

        fig_radar = go.Figure()

        for r in results_sorted[:5]:
            skill_scores = []

            for skill in skills:
                found = next((m for m in r["matched"] if m["skill"] == skill), None)
                skill_scores.append(found["score"] if found else 0)

            fig_radar.add_trace(go.Scatterpolar(
                r=skill_scores + [skill_scores[0]],
                theta=skills + [skills[0]],
                fill="toself",
                name=r["short"]
            ))

        fig_radar.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            polar=dict(
                bgcolor="rgba(15,23,42,0.75)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor="rgba(255,255,255,0.12)"
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.10)"
                )
            ),
            legend=dict(font=dict(color="white")),
            margin=dict(t=35, b=35, l=35, r=35)
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<p class="sec-title">📋 Skill Comparison Matrix</p>', unsafe_allow_html=True)

    table_data = {"Skill": skills}

    for r in results_sorted:
        values = []
        for skill in skills:
            m = next((x for x in r["matched"] if x["skill"] == skill), None)
            if m:
                icon = "🔑" if m["type"] == "keyword" else "🧠"
                values.append(f"{icon} {m['score']}%")
            else:
                values.append("❌")
        table_data[r["short"]] = values

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<p class="sec-title">📄 Detailed Candidate Analysis</p>', unsafe_allow_html=True)

    for idx, r in enumerate(results_sorted):
        medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "📄"
        color = score_color(r["pct"])

        with st.expander(
            f"{medal} {r['name']} — {r['score']}/{r['total']} matched ({r['pct']}%)",
            expanded=(idx == 0)
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                fig_donut = go.Figure(go.Pie(
                    values=[r["score"], r["total"] - r["score"]],
                    labels=["Matched", "Missing"],
                    hole=0.70,
                    marker=dict(
                        colors=[color, "rgba(255,255,255,0.10)"],
                        line=dict(color="rgba(255,255,255,0.18)", width=2)
                    ),
                    textinfo="none"
                ))

                fig_donut.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        y=-0.10,
                        font=dict(color="white")
                    ),
                    annotations=[dict(
                        text=f"<b>{r['pct']}%</b>",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=30, color=color)
                    )],
                    margin=dict(t=10, b=10, l=10, r=10)
                )

                st.plotly_chart(fig_donut, use_container_width=True)

                keyword_count = sum(1 for m in r["matched"] if m["type"] == "keyword")
                semantic_count = sum(1 for m in r["matched"] if m["type"] == "semantic")

                st.markdown(f"🔑 Keyword Matches: **{keyword_count}**")
                st.markdown(f"🧠 Semantic Matches: **{semantic_count}**")
                st.markdown(f"❌ Missing Skills: **{len(r['missing'])}**")

            with col2:
                st.markdown(f"### ✅ Matched Skills ({len(r['matched'])})")

                for m in r["matched"]:
                    css_class = "skill-match-keyword" if m["type"] == "keyword" else "skill-match"
                    icon = "🔑" if m["type"] == "keyword" else "🧠"

                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            {icon} <b>{m['skill']}</b> — {m['score']}%
                            <br>
                            <span style="opacity:0.70;font-size:0.82rem;">{m['reason']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if r["missing"]:
                st.markdown(f"### ❌ Missing Skills ({len(r['missing'])})")
                miss_cols = st.columns(3)

                for j, m in enumerate(r["missing"]):
                    with miss_cols[j % 3]:
                        st.markdown(
                            f'<div class="skill-miss">✘ {m["skill"]} — {m["score"]}% similarity</div>',
                            unsafe_allow_html=True
                        )

            report = f"AI RESUME ANALYSIS REPORT\n{'=' * 55}\n\n"
            report += f"Candidate: {r['name']}\n"
            report += f"Score: {r['score']}/{r['total']} skills\n"
            report += f"Percentage: {r['pct']}%\n\n"

            report += "MATCHED SKILLS\n"
            report += "-" * 55 + "\n"

            for m in r["matched"]:
                match_type = "KEYWORD" if m["type"] == "keyword" else "SEMANTIC"
                report += f"[{match_type}] {m['skill']} — {m['score']}%\n"
                report += f"Reason: {m['reason']}\n\n"

            report += "\nMISSING SKILLS\n"
            report += "-" * 55 + "\n"

            for m in r["missing"]:
                report += f"{m['skill']} — {m['score']}% similarity\n"

            st.download_button(
                f"⬇️ Download {r['name']} Report",
                report,
                file_name=f"{r['name']}_analysis_report.txt",
                key=f"download_{idx}"
            )