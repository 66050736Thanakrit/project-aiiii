"""app.py — หน้าเว็บ Dashboard (Streamlit): เรียกใช้ data.py / models.py / ai.py"""
import random
import html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ai import ask_ai, build_context
from data import clean_sales, daily_sales, generate_sample, overall_kpis, product_summary
from models import FEATURES, regression_forecast, timeseries_forecast, train_regression

st.set_page_config(page_title="RetailMind AI", page_icon="🛒", layout="wide")

MINT, AMBER, CORAL, VIOLET, TEXT = "#0e9f6e", "#d97706", "#e11d48", "#6d5bd0", "#0f172a"
DARK = st.session_state.get("dark_mode", False)
FG, GL, GG = ("#e2e8f0", "rgba(255,255,255,.25)", "rgba(255,255,255,.1)") if DARK else ("#1e293b", "rgba(15,23,42,.3)", "rgba(15,23,42,.1)")
if DARK:
    TEXT = "#e8edf5"


# ---------------------------------------------------------------
# ธีม + พื้นหลังขยับได้ (CSS ล้วน): แท่งกราฟยอดขายขยับขึ้นลง + ไอคอนร้านค้าลอย + แสงสีเคลื่อนที่
# ---------------------------------------------------------------
def background() -> str:
    rnd = random.Random(3)
    bars = "".join(
        f'<i style="height:{rnd.randint(25, 100)}%;animation-duration:{rnd.uniform(4, 9):.1f}s;'
        f'animation-delay:-{rnd.uniform(0, 9):.1f}s"></i>' for _ in range(40))
    icons = "".join(
        f'<b style="left:{rnd.randint(2, 96)}%;font-size:{rnd.randint(20, 36)}px;'
        f'animation-duration:{rnd.randint(18, 34)}s;animation-delay:-{rnd.randint(0, 30)}s">{e}</b>'
        for e in ["🛒", "☕", "🥐", "🧾", "📦", "🏷️", "🍰", "💳"] * 2)
    return f'<div class="bgbars">{bars}</div><div class="bgicons">{icons}</div><div class="blob b1"></div><div class="blob b2"></div>'


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
:root {{ --card:rgba(255,255,255,.86); --ln:rgba(15,23,42,.14); --mu:#475569; --ink:#0f172a; }}
html, body, [class*="css"] {{ font-family:'Prompt',sans-serif !important; }}
.stApp {{ background:radial-gradient(1100px 600px at 88% -8%,rgba(109,91,208,.14),transparent 60%),#f3f6fb; color:var(--ink); }}
/* ตัวหนังสือสีเข้ม อ่านง่าย */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color:var(--ink); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:#334155 !important; }}
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{ position:relative; z-index:2; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stDecoration"], #MainMenu, footer {{ display:none !important; }}
[data-testid="stToolbar"] {{ visibility:hidden; }}
/* ปุ่มเปิด sidebar กลับ (ไม่งั้นพอยุบแล้วจะหาไม่เจอ) */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{
  visibility:visible !important; display:flex !important; position:fixed; top:14px; left:14px; z-index:1000002;
  padding:6px 10px; border-radius:12px; background:linear-gradient(135deg,{MINT},#0b7a8f);
  box-shadow:0 4px 0 #0a6a52, 0 8px 16px rgba(14,159,110,.3); transition:transform .2s cubic-bezier(.34,1.56,.64,1); }}
[data-testid="stExpandSidebarButton"]:hover, [data-testid="stSidebarCollapsedControl"]:hover {{ transform:translateY(-2px) scale(1.06); }}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] * {{ color:#fff !important; fill:#fff !important; }}
[role="tabpanel"] {{ outline:none !important; }}
[data-testid="stSidebar"] {{ background:rgba(255,255,255,.88) !important; border-right:1px solid var(--ln); backdrop-filter:blur(14px); }}
.block-container {{ max-width:1360px; padding-top:1.2rem; }}
[data-baseweb="select"] > div, textarea, input {{ background:#fff !important; color:var(--ink) !important; }}

/* ---- background layers ---- */
.bgbars {{ position:fixed; left:0; right:0; bottom:0; height:55vh; display:flex; align-items:flex-end; gap:6px; padding:0 6px; z-index:0; pointer-events:none; }}
.bgbars i {{ flex:1; transform-origin:bottom; border-radius:6px 6px 0 0; background:linear-gradient(180deg,rgba(14,159,110,.26),rgba(109,91,208,.04));
  animation:pulse ease-in-out infinite alternate; }}
@keyframes pulse {{ from{{transform:scaleY(.25)}} to{{transform:scaleY(1)}} }}
.bgicons {{ position:fixed; inset:0; z-index:1; pointer-events:none; overflow:hidden; }}
.bgicons b {{ position:absolute; bottom:-60px; opacity:0; animation:rise linear infinite; }}
@keyframes rise {{ 0%{{transform:translateY(0) rotate(0);opacity:0}} 12%{{opacity:.4}} 100%{{transform:translateY(-115vh) rotate(25deg);opacity:0}} }}
.blob {{ position:fixed; border-radius:50%; filter:blur(90px); opacity:.22; z-index:0; pointer-events:none; animation:drift 18s ease-in-out infinite alternate; }}
.b1 {{ width:46vw; height:46vw; left:-12vw; top:-14vw; background:{VIOLET}; }}
.b2 {{ width:40vw; height:40vw; right:-10vw; bottom:-12vw; background:{MINT}; animation-delay:-8s; }}
@keyframes drift {{ to{{transform:translate(8vw,6vh) scale(1.2)}} }}

/* ---- components ---- */
.topbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; padding:16px 24px; margin-bottom:18px;
  border:1px solid var(--ln); border-radius:18px; background:var(--card); backdrop-filter:blur(14px); box-shadow:0 4px 20px rgba(15,23,42,.07); }}
.logo {{ font-weight:700; font-size:1.35rem; letter-spacing:.06em; background:linear-gradient(90deg,{MINT},{AMBER},{VIOLET},{MINT});
  background-size:250% auto; -webkit-background-clip:text; background-clip:text; color:transparent; animation:shine 7s linear infinite; }}
.logo small {{ display:block; font-size:.62rem; letter-spacing:.3em; color:var(--mu); -webkit-text-fill-color:var(--mu); }}
@keyframes shine {{ to{{background-position:250% center}} }}
.tag {{ color:var(--mu); font-size:.85rem; }} .tag b {{ color:var(--ink); }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{MINT}; margin-right:7px; box-shadow:0 0 10px {MINT}; animation:blink 2s infinite; }}
@keyframes blink {{ 50%{{opacity:.3}} }}
.label {{ font-size:.8rem; color:#b45309; font-weight:600; letter-spacing:.14em; text-transform:uppercase; margin:18px 0 8px; }}
.note {{ padding:12px 16px; border-left:3px solid {MINT}; background:rgba(14,159,110,.1); border-radius:10px; margin:10px 0; color:var(--ink); }}
div[data-testid="stMetric"] {{ background:var(--card); border:1px solid var(--ln); border-radius:16px; padding:16px 20px; backdrop-filter:blur(12px);
  box-shadow:0 4px 16px rgba(15,23,42,.06); transition:.25s; }}
div[data-testid="stMetric"]:hover {{ transform:translateY(-3px); border-color:{MINT}; box-shadow:0 10px 28px rgba(14,159,110,.2); }}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{ color:#334155 !important; font-weight:500; letter-spacing:.04em; }}
div[data-testid="stMetricValue"] {{ color:var(--ink) !important; font-weight:700; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--card); border-color:var(--ln) !important; border-radius:16px; backdrop-filter:blur(12px); }}
/* tabs: ดูบล็อกสไตล์แท็บด้านล่าง */
.stButton > button, .stDownloadButton > button {{ border-radius:10px; border:1px solid var(--ln); background:#fff; transition:.2s; }}
.stButton > button[kind="primary"] {{ background:linear-gradient(135deg,{MINT},#0b7a8f); border:0; font-weight:600; }}
.stButton > button[kind="primary"] p {{ color:#ffffff !important; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ box-shadow:0 0 18px rgba(14,159,110,.35); border-color:{MINT}; }}
</style>
{background()}
""", unsafe_allow_html=True)

st.markdown("""
<style>
[role="tablist"] { gap:16px !important; padding:10px 6px 20px !important; border-bottom:0 !important;
  background:transparent !important; box-shadow:none !important; overflow-x:auto; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; height:0 !important; background:transparent !important; }

[role="tab"] { position:relative; overflow:hidden; height:auto !important; padding:15px 32px !important;
  background:#ffffff !important; border:2.5px solid rgba(14,159,110,.28) !important; border-radius:20px !important;
  box-shadow:0 4px 0 rgba(14,159,110,.18) !important; outline:none !important;
  transition:transform .2s cubic-bezier(.34,1.56,.64,1), box-shadow .2s, border-color .2s, background .2s !important; }
[role="tab"] * { color:#334155 !important; font-weight:500; white-space:nowrap; font-size:1.18rem !important; line-height:1.3; }

[role="tab"]:hover { transform:translateY(-4px) rotate(-1deg); border-color:#0e9f6e !important;
  background:#f0fdf8 !important; box-shadow:0 8px 0 rgba(14,159,110,.22) !important; }
[role="tab"]:active { transform:translateY(2px) scale(.96); box-shadow:0 1px 0 rgba(14,159,110,.2) !important; }
[role="tab"]:focus-visible { box-shadow:0 0 0 3px rgba(14,159,110,.4) !important; }

[role="tab"][aria-selected="true"] { background:linear-gradient(135deg,#0e9f6e,#0b7a8f) !important; border-color:#0a6a52 !important;
  box-shadow:0 5px 0 #0a6a52, 0 12px 22px rgba(14,159,110,.3) !important; transform:translateY(-3px);
  animation:tabpop .45s cubic-bezier(.34,1.56,.64,1); }
[role="tab"][aria-selected="true"] * { color:#ffffff !important; font-weight:600; }
[role="tab"][aria-selected="true"]::after { content:""; position:absolute; top:0; left:-60%; width:40%; height:100%;
  background:linear-gradient(100deg,transparent,rgba(255,255,255,.5),transparent); transform:skewX(-20deg);
  animation:sheen .9s ease .1s 1 both; pointer-events:none; }
@keyframes tabpop { 0%{transform:scale(.85)} 60%{transform:scale(1.07) translateY(-3px)} 100%{transform:scale(1) translateY(-3px)} }
@keyframes sheen { from{left:-60%} to{left:140%} }

[role="tabpanel"] { animation:tabin .4s ease both; }
@keyframes tabin { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
@media (prefers-reduced-motion:reduce) { [role="tab"], [role="tab"]::after, [role="tabpanel"] { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(255,255,255,.94),rgba(230,250,241,.92)) !important; }
.sb-brand { display:flex; gap:12px; align-items:center; padding:14px 16px; margin-bottom:14px; border-radius:16px;
  background:linear-gradient(135deg,#0e9f6e,#0b7a8f); box-shadow:0 8px 20px rgba(14,159,110,.3); }
.sb-brand b { display:block; font-size:1.05rem; color:#fff; }
.sb-brand small { color:rgba(255,255,255,.85); }
.sb-emoji { font-size:1.9rem; display:inline-block; animation:sbfloat 3s ease-in-out infinite; }
@keyframes sbfloat { 50% { transform:translateY(-5px) rotate(-6deg); } }
.sb-title { font-weight:600; margin:18px 0 8px; color:#0f172a; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed rgba(14,159,110,.45) !important; border-radius:14px !important;
  background:rgba(255,255,255,.75) !important; transition:transform .2s, border-color .2s, background .2s; }
[data-testid="stFileUploaderDropzone"]:hover { transform:translateY(-2px); border-color:#0e9f6e !important; background:#f0fdf8 !important; }
.sb-card { background:#fff; border:1px solid rgba(15,23,42,.1); border-radius:14px; padding:12px 14px; margin-bottom:10px;
  box-shadow:0 3px 10px rgba(15,23,42,.05); transition:transform .2s, box-shadow .2s, border-color .2s; }
.sb-card:hover { transform:translateY(-3px); border-color:#0e9f6e; box-shadow:0 10px 22px rgba(14,159,110,.18); }
.sb-k { font-size:.75rem; color:#475569; }
.sb-v { font-size:1.25rem; font-weight:700; color:#0f172a; }
.sb-row { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.sb-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px; }
.sb-badge { padding:2px 9px; border-radius:99px; font-size:.75rem; font-weight:600; }
.sb-badge.up { background:#d9f5ea; color:#0a6a52; }
.sb-badge.down { background:#ffe0ea; color:#be123c; }
.sb-spark polyline { stroke-dasharray:1; stroke-dashoffset:1; animation:sbdraw 1.4s ease forwards; }
@keyframes sbdraw { to { stroke-dashoffset:0; } }
.sb-status { display:flex; align-items:center; gap:8px; font-size:.82rem; color:#0f172a; }
.sb-dot { width:9px; height:9px; border-radius:50%; background:#0e9f6e; box-shadow:0 0 8px #0e9f6e; animation:blink 2s infinite; }
.sb-dot.off { background:#e11d48; box-shadow:0 0 8px #e11d48; animation:none; }
.sb-chips { display:flex; flex-wrap:wrap; gap:6px; }
.sb-chip { padding:5px 11px; border-radius:99px; background:#fff; border:1px solid rgba(14,159,110,.3); font-size:.78rem; color:#0f172a;
  transition:transform .2s cubic-bezier(.34,1.56,.64,1), background .2s; }
.sb-chip:hover { transform:translateY(-2px) scale(1.06); background:#d9f5ea; }
@media (prefers-reduced-motion:reduce) { .sb-emoji, .sb-spark polyline, .sb-dot { animation:none !important; stroke-dashoffset:0; } }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# ฟังก์ชันช่วย
# ---------------------------------------------------------------
def baht(v): return "N/A" if pd.isna(v) else f"฿{v:,.0f}"
def pct(v): return "N/A" if pd.isna(v) else f"{v:+.1f}%"
def label(t): st.markdown(f'<div class="label">{t}</div>', unsafe_allow_html=True)


def draw(fig, h=400, **kw):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=FG, family="Prompt", size=13), margin=dict(l=10, r=10, t=10, b=10),
                      hovermode=kw.pop("hover", "x unified"), legend=dict(orientation="h", y=1.12),
                      xaxis=dict(showgrid=False, linecolor=GL), yaxis=dict(gridcolor=GG, zeroline=False), **kw)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


st.markdown("""
<style>
html { scroll-behavior:smooth; }
:root { --panel:rgba(255,255,255,.80); --chart:rgba(255,255,255,.96); }
.block-container { background:var(--panel); backdrop-filter:blur(16px) saturate(1.1); border:1px solid var(--ln);
  border-radius:24px; box-shadow:0 10px 40px rgba(15,23,42,.10); padding:1.6rem 2rem 3rem !important; margin-top:.8rem; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background:var(--chart); border:1px solid var(--ln);
  border-radius:16px; padding:8px; box-shadow:0 3px 14px rgba(15,23,42,.06); }
.stApp { transition:background-color .4s, color .4s; }
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:linear-gradient(#0e9f6e,#0b7a8f); border-radius:99px; }
::selection { background:rgba(14,159,110,.3); }
div[data-testid="stMetric"] { border-left:4px solid #0e9f6e !important; }
[data-testid="stChatMessage"] { border-radius:16px; border:1px solid var(--ln); background:var(--card); margin-bottom:8px; transition:transform .2s; }
[data-testid="stChatMessage"]:hover { transform:translateX(3px); }
</style>
""", unsafe_allow_html=True)

if DARK:
    st.markdown("""
<style>
:root { --card:rgba(20,28,48,.88); --ln:rgba(255,255,255,.14); --mu:#a8b3c7; --ink:#e8edf5; --panel:rgba(11,18,32,.82); --chart:rgba(17,24,43,.96); }
.stApp { background:radial-gradient(1100px 600px at 88% -8%,rgba(109,91,208,.25),transparent 60%),#0b1220 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color:#a8b3c7 !important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(15,23,42,.96),rgba(10,26,28,.94)) !important; }
[data-baseweb="select"] > div, textarea, input { background:#16213a !important; color:var(--ink) !important; }
.stButton > button, .stDownloadButton > button { background:#16213a; color:var(--ink); border-color:var(--ln); }
.stButton > button p, .stDownloadButton > button p { color:var(--ink); }
[role="tab"] { background:#16213a !important; border-color:rgba(255,255,255,.14) !important; box-shadow:0 4px 0 rgba(0,0,0,.35) !important; }
[role="tab"] * { color:#cbd5e1 !important; }
[role="tab"][aria-selected="false"]:hover { background:#1d2c4d !important; }
.sb-card, .sb-chip { background:#16213a; border-color:rgba(255,255,255,.14); }
.sb-title, .sb-v, .sb-status, .sb-chip { color:var(--ink); }
.sb-k { color:#9fb0c8; }
.sb-badge.up { background:rgba(14,159,110,.25); color:#6ee7b7; }
.sb-badge.down { background:rgba(225,29,72,.25); color:#fda4af; }
[data-testid="stFileUploaderDropzone"] { background:rgba(22,33,58,.8) !important; }
[data-testid="stDataFrame"] { filter:invert(.92) hue-rotate(180deg); }
pre, [data-testid="stCode"] pre { background:#0f1a30 !important; }
pre code, [data-testid="stCode"] code { color:#e2e8f0 !important; }
.note { background:rgba(14,159,110,.16); }
</style>
""", unsafe_allow_html=True)


if not st.session_state.get("bg_anim", True):
    st.markdown("<style>.bgbars, .bgicons, .blob { display:none !important; }</style>", unsafe_allow_html=True)


@st.cache_data
def sample_data(): return generate_sample()
@st.cache_data
def fit_regression(daily): return train_regression(daily)
@st.cache_data
def fit_ts(daily, horizon): return timeseries_forecast(daily, horizon)


def find_anomalies(daily, z=4.0):
    """วันที่ยอดผิดปกติ: ปรับตามวันในสัปดาห์ก่อน แล้ววัดห่างจากค่ากลางเคลื่อนที่ด้วย robust z-score (MAD)"""
    s = daily.set_index("Date")["Sales"]
    adj = s / s.groupby(s.index.dayofweek).transform("mean").replace(0, np.nan)
    r = adj - adj.rolling(28, center=True, min_periods=7).median()
    mad = float((r - r.median()).abs().median()) * 1.4826
    zs = r / (mad if mad > 0 else 1)
    return daily.assign(Z=zs.values)[(zs.abs() > z).values].reset_index(drop=True)


def health_score(kpi, products, daily):
    """คะแนนสุขภาพร้านค้า 0-100 (ประมาณการ): โต 40% · นิ่ง 30% · สินค้าหลากหลาย 30%"""
    g = 0 if pd.isna(kpi["growth30"]) else kpi["growth30"]
    growth = float(np.clip(50 + g * 2.5, 0, 100))
    cv = daily["Sales"].std() / max(daily["Sales"].mean(), 1)
    stable = float(np.clip(100 - cv * 150, 0, 100))
    diverse = float(np.clip(100 - products["Share"].max() * 1.5, 0, 100))
    return round(0.4 * growth + 0.3 * stable + 0.3 * diverse)


# ---------------------------------------------------------------
# Sidebar + โหลดข้อมูล
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-brand"><span class="sb-emoji">🛒</span>'
                '<div><b>RetailMind AI</b><small>ผู้ช่วยวิเคราะห์ร้านค้า</small></div></div>', unsafe_allow_html=True)
    st.toggle("🌙 โหมดมืด", key="dark_mode")
    st.toggle("🎬 พื้นหลังเคลื่อนไหว", value=True, key="bg_anim")
    st.markdown('<div class="sb-title">📁 ข้อมูลร้านค้า</div>', unsafe_allow_html=True)
    upload = st.file_uploader("อัปโหลดยอดขาย (.csv / .xlsx)", type=["csv", "xlsx"])
    st.caption("ต้องมี: Product, Quantity, Price · แนะนำ: Date, Category, Promotion")

source, raw = "ข้อมูลตัวอย่าง (จำลอง)", sample_data()
if upload is not None:
    try:
        raw = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
        source = upload.name
        if st.session_state.get("seen_file") != upload.name:
            st.session_state["seen_file"] = upload.name
            st.toast(f"โหลดไฟล์ {upload.name} สำเร็จ", icon="✅")
            st.balloons()
    except Exception as e:
        st.error(f"อ่านไฟล์ไม่ได้: {e}")
try:
    df = clean_sales(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.markdown('<div class="sb-title">🎛️ ตัวกรอง</div>', unsafe_allow_html=True)
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    rng = st.date_input("ช่วงวันที่", (dmin, dmax), min_value=dmin, max_value=dmax)
    cats = st.multiselect("หมวดหมู่", sorted(df["Category"].astype(str).unique()))
    prods = st.multiselect("สินค้า", sorted(df["Product"].astype(str).unique()))
if isinstance(rng, (tuple, list)) and len(rng) == 2:
    df = df[(df["Date"] >= pd.Timestamp(rng[0])) & (df["Date"] <= pd.Timestamp(rng[1]))]
if cats:
    df = df[df["Category"].astype(str).isin(cats)]
if prods:
    df = df[df["Product"].astype(str).isin(prods)]
if df["Date"].nunique() < 60:
    st.warning("หลังกรองเหลือข้อมูลน้อยกว่า 60 วัน โมเดลพยากรณ์ต้องใช้อย่างน้อย 60 วัน กรุณาขยายตัวกรอง")
    st.stop()

daily = daily_sales(df)
anom = find_anomalies(daily)
kpi = overall_kpis(daily)
products = product_summary(df)
reg = fit_regression(daily)

def spark(vals, color="#0e9f6e", w=200, h=40):
    v = list(vals)
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 3 - (x - lo) / rng * (h - 8):.1f}" for i, x in enumerate(v))
    return (f'<svg class="sb-spark" viewBox="0 0 {w} {h}" width="100%"><polyline points="{pts}" pathLength="1" fill="none" '
            f'stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')


with st.sidebar:
    g = kpi["growth30"]
    badge = ("" if pd.isna(g) else f'<span class="sb-badge {"up" if g >= 0 else "down"}">{"▲" if g >= 0 else "▼"} {abs(g):.1f}%</span>')
    st.markdown(f"""
    <div class="sb-title">📊 สรุปข้อมูล</div>
    <div class="sb-card"><div class="sb-grid">
      <div><div class="sb-k">แถวข้อมูล</div><div class="sb-v">{len(df):,}</div></div>
      <div><div class="sb-k">จำนวนวัน</div><div class="sb-v">{kpi['days']:,}</div></div>
      <div><div class="sb-k">สินค้า</div><div class="sb-v">{df['Product'].nunique():,}</div></div>
      <div><div class="sb-k">หมวดหมู่</div><div class="sb-v">{df['Category'].nunique():,}</div></div>
    </div></div>
    <div class="sb-card">
      <div class="sb-row"><span class="sb-k">ยอดขาย 30 วันล่าสุด</span>{badge}</div>
      <div class="sb-v">{baht(kpi['last30'])}</div>
      {spark(daily['Sales'].tail(30))}
    </div>""", unsafe_allow_html=True)

    top = products.iloc[0]
    rising = products.dropna(subset=["Growth30d"]).sort_values("Growth30d")
    rise_html = ""
    if len(rising) and rising.iloc[-1]["Growth30d"] > 0:
        r = rising.iloc[-1]
        rise_html = (f'<div class="sb-card"><div class="sb-k">🚀 โตเร็วสุด (30 วัน)</div>'
                     f'<div class="sb-row"><span class="sb-v" style="font-size:1rem">{html.escape(str(r["Product"]))}</span>'
                     f'<span class="sb-badge up">{r["Growth30d"]:+.1f}%</span></div></div>')
    st.markdown(f"""
    <div class="sb-title">⭐ สินค้าเด่น</div>
    <div class="sb-card"><div class="sb-k">👑 ขายดีสุด</div>
      <div class="sb-row"><span class="sb-v" style="font-size:1rem">{html.escape(str(top['Product']))}</span>
      <span class="sb-badge up">{top['Share']:.1f}%</span></div></div>
    {rise_html}""", unsafe_allow_html=True)

    ai_ok = bool(os.getenv("GEMINI_API_KEY"))
    st.markdown(f"""
    <div class="sb-title">⚙️ ระบบ</div>
    <div class="sb-card"><div class="sb-status"><span class="sb-dot {'' if ai_ok else 'off'}"></span>
      {'Gemini พร้อมใช้งาน' if ai_ok else 'ยังไม่ได้ตั้งค่า GEMINI_API_KEY'}</div></div>
    <div class="sb-chips"><span class="sb-chip">📐 Ridge</span><span class="sb-chip">🌲 Random Forest</span>
      <span class="sb-chip">📈 Holt-Winters</span><span class="sb-chip">✨ Gemini</span></div>
    """, unsafe_allow_html=True)
    st.write("")
    st.download_button("⬇️ ดาวน์โหลดสรุปรายวัน (CSV)", daily.to_csv(index=False).encode("utf-8-sig"),
                       file_name="daily_sales.csv", mime="text/csv", width="stretch")

st.markdown(f"""<div class="topbar"><div class="logo">RETAILMIND AI<small>SALES ANALYTICS · FORECAST · GEN-AI INSIGHTS</small></div>
<div class="tag"><span class="dot"></span>ข้อมูล: <b>{source}</b> · {kpi['start']} → {kpi['end']}</div></div>""", unsafe_allow_html=True)

tabs = st.tabs(["📊 ภาพรวม", "🔍 ปัจจัยที่มีผล", "📦 สินค้า", "🧪 โมเดล Regression", "📈 พยากรณ์", "🧠 Gen-AI"])

# ---------- 1) ภาพรวม ----------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("ยอดขายรวม", baht(kpi["total"]))
    c[1].metric("เฉลี่ยต่อวัน", baht(kpi["avg_daily"]))
    c[2].metric("30 วันล่าสุด", baht(kpi["last30"]), pct(kpi["growth30"]))
    c[3].metric("จำนวนชิ้นที่ขาย", f"{kpi['qty']:,.0f}")

    g1, g2 = st.columns([1, 1.6])
    with g1:
        label("สุขภาพร้านค้า")
        gf = go.Figure(go.Indicator(mode="gauge+number", value=health_score(kpi, products, daily), number=dict(suffix=" / 100", font=dict(size=34)),
                                    gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=MINT),
                                               steps=[dict(range=[0, 40], color="rgba(225,29,72,.18)"),
                                                      dict(range=[40, 70], color="rgba(217,119,6,.18)"),
                                                      dict(range=[70, 100], color="rgba(14,159,110,.18)")])))
        gf.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Prompt", color=FG))
        st.plotly_chart(gf, width="stretch", config={"displayModeBar": False})
        st.caption("คะแนนประมาณการ: การเติบโต 40% · ความนิ่งของยอด 30% · ความหลากหลายสินค้า 30%")
    with g2:
        label("วันที่ยอดผิดปกติ")
        if len(anom):
            st.markdown(f'<div class="note">พบ <b>{len(anom)}</b> วันที่ยอดพุ่งหรือตกผิดปกติ (ปรับตามวันในสัปดาห์แล้ว)</div>', unsafe_allow_html=True)
            show = pd.DataFrame({"วันที่": anom["Date"].dt.strftime("%Y-%m-%d"), "ยอดขาย": anom["Sales"].round(0),
                                 "ประเภท": np.where(anom["Z"] > 0, "📈 พุ่ง", "📉 ตก")})
            st.dataframe(show.tail(6), hide_index=True, width="stretch")
        else:
            st.markdown('<div class="note">ไม่พบวันที่ยอดผิดปกติ ✅</div>', unsafe_allow_html=True)

    label("แนวโน้มยอดขาย")
    period = st.pills("ช่วงเวลา", ["7 วัน", "30 วัน", "90 วัน", "ทั้งหมด"], default="ทั้งหมด", key="period") or "ทั้งหมด"
    d = daily.assign(MA7=daily["Sales"].rolling(7).mean(), MA30=daily["Sales"].rolling(30).mean())
    n = {"7 วัน": 7, "30 วัน": 30, "90 วัน": 90}.get(period)
    d = d.tail(n) if n else d
    fig = go.Figure()
    fig.add_scatter(x=d["Date"], y=d["Sales"], name="ยอดจริง", line=dict(color="rgba(71,85,105,.5)", width=1))
    fig.add_scatter(x=d["Date"], y=d["MA7"], name="เฉลี่ย 7 วัน", line=dict(color=MINT, width=2))
    fig.add_scatter(x=d["Date"], y=d["MA30"], name="แนวโน้ม 30 วัน", line=dict(color=AMBER, width=2.6))
    a_in = anom[anom["Date"].isin(d["Date"])]
    if len(a_in):
        fig.add_scatter(x=a_in["Date"], y=a_in["Sales"], mode="markers", name="ผิดปกติ",
                        marker=dict(color=CORAL, size=11, symbol="circle-open", line=dict(width=2.5)))
    draw(fig, 420)

    c1, c2 = st.columns(2)
    with c1:
        label("ยอดขายรายเดือน")
        mp = st.pills("ช่วงเดือน", ["6 เดือน", "12 เดือน", "ทั้งหมด"], default="ทั้งหมด", key="mp", label_visibility="collapsed") or "ทั้งหมด"
        m = daily.set_index("Date")["Sales"].resample("MS").sum().reset_index()
        m = m.tail({"6 เดือน": 6, "12 เดือน": 12}.get(mp) or len(m))
        draw(go.Figure(go.Bar(x=m["Date"], y=m["Sales"], marker_color=VIOLET)), 320)
    with c2:
        label("ยอดเฉลี่ยตามวันในสัปดาห์")
        w = daily.groupby(daily["Date"].dt.dayofweek)["Sales"].mean()
        draw(go.Figure(go.Bar(x=["จ.", "อ.", "พ.", "พฤ.", "ศ.", "ส.", "อา."], y=w.values,
                              marker_color=[MINT if v == w.max() else "rgba(14,159,110,.4)" for v in w.values])), 320)

# ---------- 2) ปัจจัย ----------
with tabs[1]:
    if reg is None:
        st.info("ข้อมูลน้อยเกินไปสำหรับวิเคราะห์ปัจจัย (ต้องมีอย่างน้อย ~50 วัน)")
    else:
        c1, c2 = st.columns(2)
        with c1:
            label(f"ความสำคัญของปัจจัย (Permutation Importance · {reg['best']})")
            imp = reg["importance"].sort_values()
            draw(go.Figure(go.Bar(x=imp.values * 100, y=[FEATURES[i] for i in imp.index], orientation="h", marker_color=AMBER)), 380, hover="closest")
        with c2:
            label("ความสัมพันธ์กับยอดขาย (Correlation)")
            cr = reg["corr"].sort_values()
            draw(go.Figure(go.Bar(x=cr.values, y=[FEATURES[i] for i in cr.index], orientation="h",
                                  marker_color=[MINT if v >= 0 else CORAL for v in cr.values])), 380, hover="closest")
        top = reg["importance"].sort_values(ascending=False)
        st.markdown(f'<div class="note">ปัจจัยที่มีอิทธิพลสูงสุดคือ <b>{FEATURES[top.index[0]]}</b> ({top.iloc[0]:.0%}) '
                    f'รองลงมา <b>{FEATURES[top.index[1]]}</b> ({top.iloc[1]:.0%})</div>', unsafe_allow_html=True)
        label("ผลของโปรโมชัน")
        a, b, c3 = st.columns(3)
        a.metric("ยอดเฉลี่ยวันปกติ", baht(kpi["normal_avg"]))
        b.metric("ยอดเฉลี่ยวันโปรโมชัน", baht(kpi["promo_avg"]))
        c3.metric("Promo Uplift", pct(kpi["uplift"]))

# ---------- 3) สินค้า ----------
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        label("Pareto 80/20 (เขียว = กลุ่ม A)")
        fig = go.Figure()
        fig.add_bar(x=products["Product"], y=products["Sales"], name="ยอดขาย",
                    marker_color=products["ABC"].map({"A": MINT, "B": AMBER, "C": "#94a3b8"}))
        fig.add_scatter(x=products["Product"], y=products["CumShare"], name="สะสม %", yaxis="y2", line=dict(color=TEXT, width=2))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False))
        draw(fig, 400)
    with c2:
        label("แผนภาพศักยภาพสินค้า (ส่วนแบ่ง × การเติบโต 30 วัน)")
        g = products["Growth30d"].fillna(0)
        fig = go.Figure(go.Scatter(
            x=products["Share"], y=g, mode="markers+text", text=products["Product"], textposition="top center",
            marker=dict(size=products["Sales"] / products["Sales"].max() * 44 + 10, color=g,
                        colorscale=[[0, CORAL], [.5, AMBER], [1, MINT]], line=dict(color=TEXT, width=1), opacity=.9)))
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(15,23,42,.35)")
        fig.add_vline(x=products["Share"].median(), line_dash="dot", line_color="rgba(15,23,42,.35)")
        fig.update_layout(xaxis_title="ส่วนแบ่งยอดขาย %", yaxis_title="เติบโต 30 วัน %")
        draw(fig, 400, hover="closest")

    c1, c2 = st.columns([2, 1])
    with c1:
        label("ตารางจัดอันดับและคำแนะนำ")
        st.dataframe(products[["Product", "Category", "Sales", "Share", "ABC", "Growth30d", "Action"]].round(1),
                     hide_index=True, width="stretch")
    with c2:
        label("ยอดขายตามหมวดหมู่")
        cat = df.groupby("Category")["Sales"].sum()
        draw(go.Figure(go.Pie(labels=cat.index, values=cat.values, hole=.58,
                              marker=dict(colors=[MINT, AMBER, VIOLET, CORAL, "#0891b2"], line=dict(color="#ffffff", width=2)))), 340)

# ---------- 4) โมเดล ----------
with tabs[3]:
    if reg is None:
        st.info("ข้อมูลน้อยเกินไปสำหรับเทรนโมเดล")
    else:
        best = reg["metrics"].set_index("Model").loc[reg["best"]]
        c = st.columns(4)
        c[0].metric("โมเดลที่ดีที่สุด", reg["best"])
        c[1].metric("MAE", baht(best["MAE"]), f"baseline {baht(reg['naive_mae'])}", delta_color="off")
        c[2].metric("RMSE", baht(best["RMSE"]))
        c[3].metric("R²", f"{best['R2']:.3f}")
        label("ทำนาย vs ยอดจริง (ชุดทดสอบ 20% ท้ายสุดตามเวลา)")
        t = reg["test"]
        fig = go.Figure()
        fig.add_scatter(x=t["Date"], y=t["Actual"], name="ยอดจริง", line=dict(color=TEXT, width=2))
        fig.add_scatter(x=t["Date"], y=t["Ridge"], name="Ridge", line=dict(color=AMBER, width=1.8))
        fig.add_scatter(x=t["Date"], y=t["RandomForest"], name="Random Forest", line=dict(color=MINT, width=2.2))
        draw(fig, 400)
        label("เปรียบเทียบโมเดล")
        st.dataframe(reg["metrics"].round(3), hide_index=True, width="stretch")
        st.caption("แบ่ง train/test ตามเวลา (ไม่สุ่ม) เลือกโมเดลจาก MAE ต่ำสุด · baseline = ใช้ยอดสัปดาห์ก่อนเป็นคำทำนาย")

# ---------- 5) พยากรณ์ ----------
with tabs[4]:
    horizon = st.pills("พยากรณ์ล่วงหน้า", [7, 14, 30, 45, 60], default=30, format_func=lambda x: f"{x} วัน", key="hz") or 30
    ts_fc, backtest = fit_ts(daily, horizon)
    rf = regression_forecast(reg["model"], daily, horizon) if reg else None

    hw = st.pills("ย้อนหลังที่แสดง", ["30 วัน", "60 วัน", "90 วัน", "ทั้งหมด"], default="90 วัน", key="hw") or "90 วัน"
    hist = daily.tail({"30 วัน": 30, "60 วัน": 60, "90 วัน": 90}.get(hw) or len(daily))
    fig = go.Figure()
    fig.add_scatter(x=hist["Date"], y=hist["Sales"], name="ยอดจริง", line=dict(color=TEXT, width=2))
    fig.add_scatter(x=ts_fc["Date"], y=ts_fc["Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ts_fc["Date"], y=ts_fc["Lower"], fill="tonexty", fillcolor="rgba(217,119,6,.16)",
                    line=dict(width=0), name="ช่วงเชื่อมั่น 95%", hoverinfo="skip")
    fig.add_scatter(x=ts_fc["Date"], y=ts_fc["Forecast"], name="Holt-Winters", line=dict(color=AMBER, width=2.6))
    if rf is not None:
        fig.add_scatter(x=rf["Date"], y=rf["Forecast"], name=f"Regression ({reg['best']})", line=dict(color=MINT, width=2, dash="dot"))
    draw(fig, 440)

    c = st.columns(3)
    c[0].metric(f"ยอดขายพยากรณ์ {horizon} วัน", baht(ts_fc["Forecast"].sum()))
    c[1].metric("เฉลี่ยต่อวัน", baht(ts_fc["Forecast"].mean()))
    if backtest:
        c[2].metric("Backtest MAE (28 วัน)", baht(backtest["mae"]), f"baseline {baht(backtest['naive_mae'])}", delta_color="off")
    if reg is not None:
        label("🎮 เครื่องจำลอง What-if")
        with st.container(border=True):
            w1, w2 = st.columns(2)
            promo_n = w1.slider("จัดโปรโมชันกี่วัน", 0, min(14, horizon), 0)
            promo_s = w2.slider("เริ่มโปรโมชันอีกกี่วัน (นับจากพรุ่งนี้)", 0, max(horizon - 1, 0), 0)
            sim = regression_forecast(reg["model"], daily, horizon, promo_start=promo_s, promo_days=promo_n)
            base_sum, sim_sum = rf["Forecast"].sum(), sim["Forecast"].sum()
            k = st.columns(3)
            k[0].metric("ยอดรวมกรณีปกติ", baht(base_sum))
            k[1].metric("ยอดรวมตามสถานการณ์", baht(sim_sum), pct((sim_sum / base_sum - 1) * 100) if base_sum > 0 else None)
            k[2].metric("ส่วนต่าง", baht(sim_sum - base_sum))
            fw = go.Figure()
            fw.add_scatter(x=rf["Date"], y=rf["Forecast"], name="กรณีปกติ", line=dict(color="rgba(71,85,105,.6)", width=2, dash="dot"))
            fw.add_scatter(x=sim["Date"], y=sim["Forecast"], name="ตามสถานการณ์", line=dict(color=MINT, width=2.6))
            draw(fw, 300)
            st.caption("ผลมาจากความสัมพันธ์ระหว่างโปรโมชันกับยอดขายที่โมเดลเรียนรู้จากข้อมูลในอดีต ไม่ใช่การรับประกันเชิงเหตุและผล "
                       "ยังไม่รวมการจำลองราคา เพราะคอลัมน์ราคาเฉลี่ยในข้อมูลแกว่งตามส่วนผสมสินค้า ไม่ใช่การปรับราคาจริง")

    out = ts_fc.rename(columns={"Forecast": "HoltWinters"})
    if rf is not None:
        out["Regression"] = rf["Forecast"].values
    st.download_button("⬇️ ดาวน์โหลดผลพยากรณ์ (CSV)", out.round({c: 0 for c in out.columns if c != "Date"}).to_csv(index=False).encode("utf-8-sig"), "forecast.csv", "text/csv")

# ---------- 6) Gen AI (แชต) ----------
with tabs[5]:
    ts30, bt30 = fit_ts(daily, 30)
    context = build_context(kpi, products, daily, reg, ts30, bt30)
    if len(anom):
        context += "\nวันผิดปกติ: " + ", ".join(f"{r.Date:%Y-%m-%d} ({'พุ่ง' if r.Z > 0 else 'ตก'} {r.Sales:,.0f})" for r in anom.tail(5).itertuples())
    PRESETS = [("📌 สรุปภาพรวม", "สรุปภาพรวมยอดขาย สาเหตุที่ยอดขายเปลี่ยนแปลง และเสนอแผนปฏิบัติ 3 ข้อเพื่อเพิ่มยอดขาย"),
               ("⭐ สินค้าเด่น", "สินค้าตัวไหนมีศักยภาพควรลงทุนเพิ่ม และตัวไหนควรทบทวน เพราะอะไร"),
               ("🎯 โปรโมชัน", "โปรโมชันได้ผลแค่ไหน และควรวางแผนโปรโมชันอย่างไร"),
               ("📦 สต็อก", "จากผลพยากรณ์ ควรเตรียมสต็อกและกำลังคนอย่างไรใน 30 วันข้างหน้า"),
               ("🔎 วันผิดปกติ", "วันที่ยอดผิดปกติอาจเกิดจากอะไรได้บ้าง และควรตรวจสอบหรือรับมืออย่างไร")]
    st.session_state.setdefault("chat", [])
    left, right = st.columns([1.6, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### 🧠 AI Retail Consultant")
            st.caption("ตัวเลขทั้งหมดคำนวณด้วยโค้ด Python แล้วส่งให้ AI สรุป — AI ไม่ได้คำนวณเอง · ถามต่อเนื่องได้")
            for col, (lbl, text) in zip(st.columns(len(PRESETS)), PRESETS):
                if col.button(lbl, key=f"preset_{lbl}", width="stretch"):
                    st.session_state["pending"] = text
            for msg in st.session_state["chat"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            prompt = st.chat_input("ถามเกี่ยวกับยอดขายของร้าน...") or st.session_state.pop("pending", None)
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                hist = "\n".join(f"{'ผู้ใช้' if m['role'] == 'user' else 'AI'}: {m['content'][:600]}" for m in st.session_state["chat"][-4:])
                full = f"บทสนทนาก่อนหน้า:\n{hist}\n\nคำถามใหม่: {prompt}" if hist else prompt
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("AI กำลังวิเคราะห์ข้อมูลยอดขาย..."):
                            ans = ask_ai(full, context)
                    except Exception as e:
                        ans = f"เชื่อมต่อ AI ไม่สำเร็จ: {e}"
                    st.markdown(ans)
                st.session_state["chat"] += [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]
            if st.session_state["chat"] and st.button("🧹 ล้างแชต"):
                st.session_state["chat"] = []
                st.rerun()
    with right:
        label("ข้อมูลสรุปที่ส่งให้ AI")
        with st.container(border=True):
            st.code(context, language=None, wrap_lines=True)


# ---------- KPI นับขึ้น (ตัวเลข st.metric ไล่ค่าจาก 0 เมื่อแสดงผล/สลับแท็บ) ----------
components.html(r"""<script>
const P = window.parent.document;
function run() {
  P.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
    if (el.offsetParent === null || el.dataset.run === '1') return;
    const n = P.createTreeWalker(el, NodeFilter.SHOW_TEXT).nextNode(); if (!n) return;
    const t = n.nodeValue; if (el.dataset.cu === t) return; el.dataset.cu = t;
    const m = t.match(/^(\D*?)([\d,]+(?:\.\d+)?)(.*)$/); if (!m) return;
    const to = parseFloat(m[2].replace(/,/g, '')), dec = (m[2].split('.')[1] || '').length, t0 = performance.now();
    el.dataset.run = '1';
    (function step(now) {
      const k = Math.min((now - t0) / 900, 1), e = 1 - Math.pow(1 - k, 3);
      n.nodeValue = m[1] + (to * e).toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec}) + m[3];
      if (k < 1) requestAnimationFrame(step); else { n.nodeValue = t; el.dataset.run = '0'; }
    })(t0);
  });
}
setInterval(run, 400); run();
</script>""", height=0)
