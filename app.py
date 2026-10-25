"""
Sage: Student Wellbeing Companion (Streamlit, v2)
Run:  python -m pip install streamlit anthropic pandas
      python -m streamlit run wellbeing_app.py
Optional: set ANTHROPIC_API_KEY (env var or .streamlit/secrets.toml) for AI replies.

A self-help tool. It is NOT a medical or therapy service.
"""
import copy
import hashlib
import hmac
import secrets
import time
from html import escape as esc
import json
import math
import os
import re
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Sage: Student Wellbeing", page_icon="🌿", layout="wide",
                   initial_sidebar_state="expanded")

MODEL = os.getenv("WELLBEING_MODEL", "claude-sonnet-4-6")
USERS_FILE = Path(__file__).with_name("sage_users.json")
DATA_DIR = Path(__file__).with_name("sage_data")

# ═════════════════════════════════════════════════════════════
# 1. SAFETY LAYER (runs before AND after the model; fails closed)
# ═════════════════════════════════════════════════════════════
HELPLINES = {
    "India": [
        ("Tele-MANAS (Govt. of India, 24x7, many languages)", "14416", "tel:14416", "Also 1-800-891-4416"),
        ("KIRAN Mental Health Helpline (Govt. of India, 24x7)", "1800-599-0019", "tel:18005990019", ""),
        ("iCall (TISS, Mumbai)", "9152987821", "tel:+919152987821", "Mon to Sat, 8am to 10pm"),
        ("AASRA (24x7)", "+91-9820466726", "tel:+919820466726", ""),
        ("Emergency services", "112", "tel:112", "If you are in immediate danger"),
    ],
    "USA": [("988 Suicide & Crisis Lifeline", "988", "tel:988", "Call or text, 24x7"),
            ("Emergency services", "911", "tel:911", "")],
    "UK": [("Samaritans", "116 123", "tel:116123", "Free, 24x7"),
           ("Emergency services", "999", "tel:999", "")],
}

# Broad on purpose: a false alarm shows a helpline; a miss could cost a life.
_PATTERNS = [
    r"\bkill(ing)?\s+my\s?self\b", r"\bkill\s+me\b", r"\bend(ing)?\s+(my\s+life|it\s+all|everything)\b",
    r"\bsuicid\w*", r"\bwant(ed)?\s+to\s+die\b", r"\bwanna\s+die\b",
    r"\b(don'?t|do\s+not)\s+want\s+to\s+(live|be\s+alive|exist|wake\s+up)\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b", r"\bbetter\s+off\s+(dead|without\s+me)\b",
    r"\bself[\s-]?harm\w*", r"\bhurt(ing)?\s+my\s?self\b", r"\bcut(ting)?\s+my\s?self\b",
    r"\bcut(ting)?\s+my\s+(wrist|arm|thigh)s?\b", r"\btake\s+my\s+(own\s+)?life\b",
    r"\boverdos\w*", r"\bhang\s+my\s?self\b",
    r"\bjump(ing)?\s+(off|from)\s+(the\s+)?(roof|building|bridge|terrace|balcony)\b",
    r"\bdisappear\s+forever\b", r"\bcan'?t\s+(go\s+on|take\s+it\s+anymore|do\s+this\s+anymore)\b",
    r"\bnobody\s+would\s+(miss|care)\b", r"\bgive\s+up\s+on\s+(life|everything)\b",
    r"\bhopeless\b.*\b(always|forever|never)\b", r"\bsevere(ly)?\s+depress\w*",
    r"\bclinical(ly)?\s+depress\w*", r"\bwish\s+i\s+(was|were)\s+(dead|never\s+born)\b",
    r"\bsleep\s+forever\b", r"\bnot\s+worth\s+living\b", r"\bworthless\b.*\b(die|dead|end)\b",
]
_RX = [re.compile(p, re.I) for p in _PATTERNS]
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s", "!": "i"})


def crisis_detected(text) -> bool:
    if not text or not isinstance(text, str):
        return False
    low = text.lower()
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s-]", " ", low.translate(_LEET)))
    compact = re.sub(r"[\s._*-]+", "", low)
    if any(k in compact for k in ("suicide", "killmyself", "endmylife", "kms", "wanttodie", "selfharm")):
        return True
    return any(rx.search(v) for rx in _RX for v in (text, norm, compact))


def render_helplines(country: str):
    st.error("💙 **I'm glad you told me, and I'm worried about you.** I'm an app and I can't give you the "
             "support you deserve right now. Please talk to a trained person **now**. You don't have to be "
             "at breaking point to call, and you don't have to go through this alone.")
    st.markdown("#### 📞 Free helplines")
    for name, number, link, note_ in HELPLINES.get(country, HELPLINES["India"]):
        st.markdown(f"**{name}**: [{number}]({link}) {note_}")
    st.warning("If you are in immediate danger, call **112** (India) or your local emergency number, or go to "
               "the nearest hospital. If you can, tell a friend, family member, roommate or your college "
               "counsellor so you're not alone tonight.")


CRISIS_CHAT_MESSAGE = ("⚠️ Your message touched on something serious, so I've paused my usual reply. "
                       "Please contact one of the helplines shown on this page right now.")


def guard(text, country) -> bool:
    """True (and helplines shown) if the text trips the safety check."""
    if crisis_detected(text):
        st.session_state.crisis = True
        render_helplines(country)
        return True
    return False


# ═════════════════════════════════════════════════════════════
# 2. EMPATHETIC PROMPT + MODEL CALL
# ═════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are "Sage", a warm, calm wellbeing companion for stressed engineering students
(exams, backlogs, labs, mini-projects, placements, peer pressure, burnout).

How you respond:
1. VALIDATE first in one or two sentences. Reflect the specific feeling or situation. Never rush to fix.
2. Then offer ONE small next step, not a list. Choose what fits:
   - Pomodoro planning: ask what is due and when, split into 25-minute focus blocks with 5-minute breaks.
   - Thought reframing (CBT): name the thought, gently spot the distortion (all-or-nothing, catastrophising,
     mind-reading, should-statements), ask for evidence for and against, offer a balanced thought.
   - Breathing/grounding: guide box breathing (4-4-4-4), 4-7-8, or 5-4-3-2-1 grounding step by step.
3. Keep replies under 120 words, plain language, kind, never preachy. At most one emoji.
4. Ask at most one question at a time. Celebrate small wins. Treat sleep, food, water and movement as part of studying well.

Hard boundaries:
- You are NOT a therapist or doctor. Never diagnose, prescribe, or recommend medication.
- Never give methods or encouragement of self-harm. Never roleplay a person in crisis.
- If self-harm, suicide or severe depression comes up, do not continue the normal flow: urge them to call a
  helpline or a trusted person. (The app also intercepts these before you answer.)
- Don't promise confidentiality. Don't claim to be human. Don't help with cheating; offer study strategies.
"""

MODES = {
    "Just listen": ("Mode: listening. Reflect feelings and ask gentle open questions. No advice unless asked.",
                    ["I'm feeling overwhelmed with everything.", "I had a bad day.", "I feel like I'm falling behind."]),
    "Plan my day": ("Mode: planning. Ask what is due, then propose 25-minute focus blocks with breaks.",
                    ["I have 3 lab records due this week.", "Exams start in 10 days.", "Help me plan today."]),
    "Reframe a thought": ("Mode: CBT reframing. Help examine one thought: evidence for/against, then a balanced thought.",
                          ["I think I'm going to fail this semester.", "Everyone is smarter than me.", "I always mess things up."]),
    "Calm me down": ("Mode: calming. Guide a breathing or grounding exercise step by step, one step per message.",
                     ["I'm panicking before my exam.", "My heart is racing.", "I can't stop overthinking."]),
    "Study coach": ("Mode: study coach. Suggest techniques (active recall, spaced repetition, Feynman, past papers). "
                    "Never write assignments for submission.",
                    ["How do I study DAA in 3 days?", "I can't focus while studying.", "How do I revise faster?"]),
}


def _api_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY")


def offline_reply(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("exam", "deadline", "submission", "backlog", "viva", "due")):
        return ("That sounds like a lot of pressure, and feeling stretched makes sense. Let's shrink it: tell me "
                "the one task due first and we'll split it into 25-minute blocks in the **Focus** page.")
    if any(w in t for w in ("fail", "stupid", "dumb", "can't", "never", "always", "smarter")):
        return ("That's a heavy thought to carry. Thoughts like this feel like facts but are often harsh "
                "predictions. Try the **Reframe** page and we'll weigh the evidence together.")
    if any(w in t for w in ("anxious", "panic", "nervous", "overwhelm", "stress", "racing")):
        return ("I'm sorry you're feeling this way. Let's slow your body first: breathe in for 4, hold 4, out 4, "
                "hold 4. The **Breathe** page will guide you.")
    return ("Thank you for telling me. I'm here with you. What's weighing on you most right now: a deadline, a "
            "worry, or just tiredness? (Add an ANTHROPIC_API_KEY for richer replies.)")


def ask_model(history, extra="") -> str:
    key = _api_key()
    if not key:
        return offline_reply(history[-1]["content"])
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history
                if m["role"] in ("user", "assistant") and not m.get("safety")]
        r = client.messages.create(model=MODEL, max_tokens=350, temperature=0.6,
                                   system=SYSTEM_PROMPT + ("\n" + extra if extra else ""), messages=msgs[-14:])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:
        return offline_reply(history[-1]["content"]) + f"\n\n_(AI service unavailable: {type(e).__name__})_"


# ═════════════════════════════════════════════════════════════
# 3. STATE + OPTIONAL LOCAL SAVE
# ═════════════════════════════════════════════════════════════
ss = st.session_state
DEFAULTS = {"moods": [], "tasks": [], "wins": [], "reframes": [], "gratitude": [], "water": {},
            "focus_log": [], "srs": [], "safety_plan": {}, "name": "", "saved": [], "custom_pins": []}


def load_into_state(data: dict):
    for k, v in DEFAULTS.items():
        ss[k] = data.get(k, copy.deepcopy(v))


def data_file():
    u = ss.get("user")
    if not u or u == "guest":
        return None
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{u}.json"


def save_data():
    f = data_file()
    if f and ss.get("persist"):
        try:
            f.write_text(json.dumps({k: ss[k] for k in DEFAULTS}, indent=1), encoding="utf-8")
        except OSError:
            pass


if "init" not in ss:
    ss.init = True
    ss.user = None
    ss.persist = False
    load_into_state({})
    ss.messages = [{"role": "assistant", "content": "Hi, I'm Sage 🌿 I'm here to help with study stress. "
                                                      "How are you feeling right now?"}]
    ss.crisis = False

TODAY = dt.date.today().isoformat()
INTERVALS = [1, 3, 7, 14, 30]


def active_days() -> set:
    d = {m["date"][:10] for m in ss.moods} | set(ss.focus_log)
    d |= {w["date"] for w in ss.wins} | {g["date"] for g in ss.gratitude}
    return d


def streak() -> int:
    days, n = active_days(), 0
    cur = dt.date.today() if TODAY in days else dt.date.today() - dt.timedelta(days=1)
    while cur.isoformat() in days:
        n += 1
        cur -= dt.timedelta(days=1)
    return n


def todays_goals():
    return {
        "Check in": any(m["date"][:10] == TODAY for m in ss.moods),
        "Focus block": ss.focus_log.count(TODAY) >= 1,
        "6 glasses of water": ss.water.get(TODAY, 0) >= 6,
        "Win or gratitude": any(w["date"] == TODAY for w in ss.wins) or any(g["date"] == TODAY for g in ss.gratitude),
    }


# ═════════════════════════════════════════════════════════════
# 4. LOOK AND FEEL
# ═════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp {background:#eefbe8;}
[data-testid="stSidebar"] {background:#dff5d6;}
.stApp, .stApp button, .stApp input, .stApp textarea {font-family:'Plus Jakarta Sans',system-ui,sans-serif;}
.stApp h1, .stApp h2, .stApp h3 {font-family:'Plus Jakarta Sans',system-ui,sans-serif; font-weight:800;
  letter-spacing:-.03em; color:#0a4a1a;}
.stApp h4 {font-family:'Plus Jakarta Sans',system-ui,sans-serif; font-weight:700; letter-spacing:-.02em;}
.block-container {max-width:900px; padding-top:1.6rem;}
.stApp button p {color:inherit;}
.stApp .stButton > button, .stApp .stDownloadButton > button,
.stApp [data-testid="stFormSubmitButton"] > button {border-radius:999px; font-weight:700; padding:.45rem 1.2rem;}
.stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"],
.stApp button[data-testid="stBaseButton-primaryFormSubmit"] {background:#ff6a13; border:1.5px solid #ff6a13; color:#fff;}
.stApp button[kind="primary"]:hover, .stApp button[data-testid="stBaseButton-primary"]:hover,
.stApp button[data-testid="stBaseButton-primaryFormSubmit"]:hover {background:#e85a05; border-color:#e85a05; color:#fff;}
.stApp button[kind="secondary"], .stApp button[data-testid="stBaseButton-secondary"] {background:transparent;
  border:1.5px solid #0a4a1a; color:#0a4a1a;}
.stApp button[kind="secondary"]:hover, .stApp button[data-testid="stBaseButton-secondary"]:hover {background:#0a4a1a; color:#fff;}
.stApp .note {border-left:4px solid #ff6a13; padding:.7rem 1rem; background:rgba(255,106,19,.10);
  border-radius:0 14px 14px 0; margin:.5rem 0 1rem 0; line-height:1.55;}
.stApp .note.lamp {border-left-color:#0a4a1a; background:rgba(10,74,26,.08);}
.stApp .pill {display:inline-block; padding:.2rem .75rem; border-radius:999px; background:#cfeec8; color:#0a4a1a;
  font-size:.85rem; font-weight:600; margin:0 .4rem .3rem 0;}
[data-testid="stChatMessage"] {border-radius:20px; padding:.85rem 1rem;}
[data-testid="stSidebar"] .stRadio label {padding:.35rem .2rem; font-size:1rem; font-weight:500;}

/* home hero: big headline, floating cards, peeking mascots */
.stApp .landing {position:relative; height:440px; max-width:1200px; margin:0 auto;}
.stApp .l-hello {position:absolute; top:0; left:50%; transform:translateX(-50%); background:#0a4a1a; color:#e6ffe0;
  padding:.3rem .95rem; border-radius:999px; font-size:.88rem; font-weight:600; z-index:1; white-space:nowrap;}
.stApp .l-title {position:absolute; top:46px; left:0; right:0; text-align:center; font-weight:800;
  font-size:clamp(2.4rem,6.2vw,5rem); line-height:1; letter-spacing:-.045em; color:#0a4a1a;}
.stApp .peek {position:absolute; bottom:0; z-index:2; line-height:0;}
.stApp .peek svg {width:100%; height:auto; display:block;}
.stApp .peek.center {left:50%; transform:translateX(-50%); width:min(40%,340px);}
.stApp .peek.pl {left:16.66%; transform:translateX(-50%); width:min(27%,250px);}
.stApp .peek.pr {left:83.33%; transform:translateX(-50%); width:min(25%,230px);}
.stApp .l-card {position:absolute; top:80px; z-index:3; text-align:center;}
.stApp .l-left {left:1%; width:150px;}
.stApp .l-right {right:1%; width:190px; background:#fff; border-radius:18px; padding:.9rem 1rem; text-align:left;
  box-shadow:0 8px 24px rgba(10,74,26,.10);}
.stApp .l-tile {position:relative; width:130px; height:112px; margin:0 auto .5rem; border-radius:22px; font-size:3.3rem;
  display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#ffb36b,#ff6a13);}
.stApp .l-dot {position:absolute; right:-8px; bottom:-8px; width:34px; height:34px; border-radius:50%; background:#0a4a1a;
  color:#fff; font-size:.95rem; display:flex; align-items:center; justify-content:center;}
.stApp .l-name {font-size:.88rem; color:#2a4a30; font-weight:500;}
.stApp .l-price {font-size:1.25rem; font-weight:800; color:#0a4a1a;}
.stApp .l-cap {font-size:.8rem; font-weight:700; color:#ff6a13; margin-bottom:.3rem;}
.stApp .l-quote {font-size:.86rem; line-height:1.45; color:#1d3a24; font-weight:500;}
@media (max-width:900px) {.stApp .l-card, .stApp .peek.pl, .stApp .peek.pr {display:none;}
  .stApp .landing {height:380px;} .stApp .peek.center {width:min(60%,300px);}}

/* three-panel band under the hero */
.stApp .st-key-band {margin-top:-1rem;}
.stApp .st-key-band [data-testid="stHorizontalBlock"] {gap:0 !important;}
.stApp .st-key-pL, .stApp .st-key-pC, .stApp .st-key-pR {min-height:190px; padding:1.5rem 1.3rem; text-align:center;
  justify-content:center; align-items:center;}
.stApp .st-key-pL, .stApp .st-key-pR {background:#a9e7a3; color:#0a4a1a;}
.stApp .st-key-pL {border-radius:0 0 0 28px;}
.stApp .st-key-pR {border-radius:0 0 28px 0;}
.stApp .st-key-pC {background:#0a4a1a;}
.stApp .st-key-pC p, .stApp .cta-title {color:#fff;}
.stApp .cta-title {font-size:1.4rem; font-weight:700; line-height:1.25; margin-bottom:.9rem;}
.stApp .st-key-pC button[data-testid="stBaseButton-secondary"], .stApp .st-key-pC button[kind="secondary"] {
  border-color:rgba(255,255,255,.7); color:#fff;}
.stApp .st-key-pC button[data-testid="stBaseButton-secondary"]:hover, .stApp .st-key-pC button[kind="secondary"]:hover {
  background:#fff; color:#0a4a1a;}
.stApp .stat .row {display:flex; align-items:center; justify-content:center; gap:.7rem;}
.stApp .stat .big {font-size:3.3rem; font-weight:800; letter-spacing:-.03em; line-height:1; color:#0a4a1a;}
.stApp .stat .star {color:#ff6a13; margin-left:.3rem; font-size:2.6rem;}
.stApp .stat .sub {margin-top:.6rem; font-size:.9rem; line-height:1.45; color:#25602f;}
.stApp .stat .avs {display:inline-flex; padding-left:8px;}
.stApp .stat .avs span {display:inline-flex; width:38px; height:38px; border-radius:50%; background:#fff;
  border:2px solid #a9e7a3; align-items:center; justify-content:center; margin-left:-8px; font-size:1.15rem;}

/* white cards */
.stApp .st-key-checkin, .stApp .st-key-water {background:#fff; border-radius:26px; padding:1.4rem 1.5rem;
  box-shadow:0 8px 28px rgba(10,74,26,.08);}
.stApp .glasses {display:flex; gap:.4rem; margin:.4rem 0 .2rem;}
.stApp .gl {width:26px; height:36px; border-radius:6px 6px 11px 11px; border:2px solid #0a4a1a;}
.stApp .gl.on {background:linear-gradient(#8fe0f5,#2aa3d6); border-color:#1c86b3;}

/* pins */
.stApp .feedhead h1 {margin:0 0 .2rem 0; padding:0; font-size:2.6rem; line-height:1.05;}
.stApp .feedhead p {margin:0 0 1rem 0; opacity:.75;}
.stApp .pin {border-radius:26px; padding:1.15rem 1.15rem 1rem; color:#10301a; margin-bottom:.4rem; break-inside:avoid;}
.stApp .pin .em {font-size:2rem; line-height:1;}
.stApp .pin h4 {margin:.55rem 0 .3rem 0; padding:0; font-size:1.15rem; line-height:1.25; color:#10301a;}
.stApp .pin p {margin:.4rem 0 0 0; font-size:.94rem; line-height:1.5; color:#2c4a34;}
.stApp .pin.quote p {font-size:1.3rem; line-height:1.3; font-weight:700; letter-spacing:-.02em; color:#10301a;}
.stApp .pin .tag {display:inline-block; margin-top:.8rem; padding:.1rem .65rem; border-radius:999px;
  background:rgba(255,255,255,.7); font-size:.78rem; font-weight:600;}
.stApp .pin.dark h4, .stApp .pin.dark p, .stApp .pin.dark.quote p {color:#effde9;}
.stApp .pin.dark .tag {background:rgba(255,255,255,.16);}
.stApp .st-key-pinsearch input {border-radius:999px; padding-left:1.1rem;}

/* sign-in art */
.stApp .auth-art {background:#cfeec8; border-radius:28px; padding:2rem 1.6rem 0; overflow:hidden;}
.stApp .auth-art .l-title {position:static; text-align:left; font-size:2.8rem;}
.stApp .auth-sub {margin:.9rem 0 0 0; color:#25602f; font-size:1rem; line-height:1.5; max-width:34ch;}
.stApp .auth-peek {width:62%; margin:1.2rem auto 0; line-height:0;}
.stApp .auth-peek svg {display:block; width:100%; height:auto;}
.stApp .auth-base {background:#0a4a1a; color:#e6ffe0; margin:0 -1.6rem; padding:1.1rem 1.6rem; text-align:center; font-weight:600;}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ── READABILITY LAYER: works in both Streamlit light and dark themes ── */
:root, .stApp {color-scheme: light;}
.stApp {background:linear-gradient(180deg,#f3fded 0%,#e6f8dd 100%) fixed; color:#14301b;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stToolbar"], [data-testid="stDecoration"] {opacity:.85;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#d9f3cf,#c7ecbb); border-right:1px solid #b4e0a8;}
[data-testid="stSidebar"] * {color:#0a4a1a;}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {fill:#0a4a1a;}

/* all text in the main area */
:where(.stApp) [data-testid="stMarkdownContainer"] p,
:where(.stApp) [data-testid="stMarkdownContainer"] li,
:where(.stApp) [data-testid="stMarkdownContainer"] span {color:#14301b;}
:where(.stApp) h1, :where(.stApp) h2, :where(.stApp) h3 {color:#0a4a1a;}
.stApp [data-testid="stWidgetLabel"] p, .stApp label p, .stApp label span {color:#0a4a1a !important; font-weight:600;}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p,
.stApp .stCaption, .stApp small {color:#3d6a47 !important; opacity:1 !important;}
.stApp [data-testid="stSliderTickBarMin"], .stApp [data-testid="stSliderTickBarMax"],
.stApp [data-testid="stSliderThumbValue"] {color:#0a4a1a !important; font-weight:700;}
.stApp [data-testid="stSlider"] [role="slider"] {background:#ff6a13 !important; box-shadow:0 0 0 4px rgba(255,106,19,.2);}
.stApp [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {background:#bfe6b6;}

/* inputs: solid white fields, dark text, green borders, orange focus */
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"] {
  background:#ffffff !important; color:#10301a !important; -webkit-text-fill-color:#10301a;
  border-color:#a6d9a0 !important; border-radius:14px !important;}
.stApp input::placeholder, .stApp textarea::placeholder {color:#7b9c82 !important; -webkit-text-fill-color:#7b9c82; opacity:1;}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within,
.stApp [data-baseweb="select"] > div:focus-within {border-color:#ff6a13 !important; box-shadow:0 0 0 3px rgba(255,106,19,.18) !important;}
.stApp [data-testid="stNumberInput"] button {background:#e3f6dc !important; color:#0a4a1a !important; border:0 !important;}
.stApp [data-baseweb="select"] svg, .stApp [data-testid="stNumberInput"] svg {fill:#0a4a1a !important;}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {background:#fff !important;}
[data-baseweb="popover"] li, [data-baseweb="menu"] li, [data-baseweb="popover"] li * {color:#10301a !important;}

/* sidebar navigation: clear pill buttons, strong active state */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {gap:.2rem;}
[data-testid="stSidebar"] .stRadio label {padding:.5rem .8rem; border-radius:14px; width:100%;
  transition:background .15s, transform .15s; cursor:pointer;}
[data-testid="stSidebar"] .stRadio label:hover {background:rgba(255,255,255,.65); transform:translateX(2px);}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {background:#0a4a1a; box-shadow:0 4px 14px rgba(10,74,26,.25);}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {color:#fff !important;}
[data-testid="stSidebar"] .stRadio label > div:first-child {display:none;}  /* hide the radio dot */
[data-testid="stSidebar"] .stRadio label p {font-size:1.02rem; font-weight:600; color:#0a4a1a !important;}
[data-testid="stSidebar"] [data-testid="stExpander"] {background:rgba(255,255,255,.55); border-radius:16px; border:1px solid #b4e0a8;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#fff !important;}

/* top-of-page radios and pills (chat modes, discover categories) */
.stApp div[role="radiogroup"][aria-orientation="horizontal"] label,
.stApp [data-testid="stPills"] button, .stApp [data-testid="stBaseButton-pills"] {
  background:#fff !important; border:1.5px solid #a6d9a0 !important; border-radius:999px !important; padding:.3rem .9rem;}
.stApp [data-testid="stBaseButton-pillsActive"] {background:#0a4a1a !important; border-color:#0a4a1a !important;}
.stApp [data-testid="stBaseButton-pillsActive"] p {color:#fff !important;}
.stApp [data-testid="stBaseButton-pills"] p {color:#0a4a1a !important;}

/* tabs, expanders, metrics, tables, alerts, chat */
.stApp [data-baseweb="tab"] {font-weight:700; color:#3d6a47 !important;}
.stApp [data-baseweb="tab"][aria-selected="true"] {color:#0a4a1a !important;}
.stApp [data-baseweb="tab-highlight"] {background:#ff6a13 !important; height:3px;}
.stApp [data-testid="stExpander"] {background:#fff; border:1px solid #cdeac6; border-radius:16px;}
.stApp [data-testid="stExpander"] summary p {color:#0a4a1a !important; font-weight:700;}
.stApp [data-testid="stMetric"] {background:#fff; border-radius:18px; padding:.8rem 1rem; box-shadow:0 4px 16px rgba(10,74,26,.07);}
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] p {color:#0a4a1a !important;}
.stApp [data-testid="stDataFrame"] {background:#fff; border-radius:16px; overflow:hidden; box-shadow:0 4px 16px rgba(10,74,26,.07);}
.stApp [data-testid="stAlert"] {border-radius:16px;}
.stApp [data-testid="stAlert"] p {color:inherit !important;}
.stApp [data-testid="stChatMessage"] {background:#fff; box-shadow:0 4px 16px rgba(10,74,26,.07); margin-bottom:.6rem;}
.stApp [data-testid="stChatMessage"] p {color:#14301b !important;}
.stApp [data-testid="stChatInput"] {background:#fff; border-radius:999px;}
.stApp [data-testid="stCheckbox"] label p, .stApp [data-testid="stToggle"] label p {color:#14301b !important; font-weight:500;}

/* buttons: stronger contrast + subtle lift */
.stApp .stButton > button, .stApp .stDownloadButton > button, .stApp [data-testid="stFormSubmitButton"] > button {
  transition:transform .12s, box-shadow .12s;}
.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {transform:translateY(-1px);
  box-shadow:0 6px 16px rgba(10,74,26,.18);}
.stApp .stDownloadButton > button {background:#fff; border:1.5px solid #0a4a1a; color:#0a4a1a;}

/* cards with soft depth, fitting the pastel palette */
.stApp .st-key-checkin, .stApp .st-key-water {border:1px solid #d6efcf; box-shadow:0 10px 30px rgba(10,74,26,.10);}
.stApp .pin {box-shadow:0 6px 18px rgba(10,74,26,.10); transition:transform .15s, box-shadow .15s;}
.stApp .pin:hover {transform:translateY(-3px); box-shadow:0 12px 26px rgba(10,74,26,.16);}
.stApp .note {color:#14301b;} .stApp .note * {color:#14301b !important;}
.stApp .gl {transition:background .2s;} .stApp .gl:not(.on) {background:#eef9ff;}
.stApp .l-title {text-shadow:0 2px 0 rgba(255,255,255,.6);}
@media (max-width:640px) {.block-container {padding:1rem .8rem;} .stApp .l-title {font-size:2.2rem;}}
</style>
""", unsafe_allow_html=True)


def note(text, lamp=False):
    st.markdown(f'<div class="note{" lamp" if lamp else ""}">{text}</div>', unsafe_allow_html=True)


def ring_svg(done: int, total: int) -> str:
    c = 2 * math.pi * 52
    off = c * (1 - (done / total if total else 0))
    return (f'<svg width="124" height="124" viewBox="0 0 124 124" role="img" aria-label="{done} of {total} goals today">'
            f'<circle cx="62" cy="62" r="52" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="10"/>'
            f'<circle cx="62" cy="62" r="52" fill="none" stroke="#ff9d42" stroke-width="10" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 62 62)"/>'
            f'<text x="62" y="62" text-anchor="middle" fill="#f7fffb" font-size="28" font-weight="700" '
            f'font-family="Plus Jakarta Sans,sans-serif">{done}/{total}</text>'
            f'<text x="62" y="82" text-anchor="middle" fill="#d9f7d0" font-size="11">goals today</text></svg>')


# ═════════════════════════════════════════════════════════════
# 4b. SIGN IN (local accounts: salted PBKDF2 hashes, per-user data files)
# ═════════════════════════════════════════════════════════════
TINTS = {"blush": ("#ffe1cc", "#a24a10"), "sky": ("#d9f0fb", "#1d5a7a"), "butter": ("#fdf2c2", "#6f5600"),
         "lilac": ("#0b4d1c", "#c9f5bf"), "sage": ("#bfeab8", "#14591f"), "peach": ("#ffd2ad", "#9a4a10")}
COLLAGE = [("blush", "🌿", "Take the next small step.", 170),
           ("sky", "🌬️", "Breathe in 4. Hold 4. Out 4. Hold 4.", 210),
           ("butter", "🍅", "25 focused minutes, then a real break.", 190),
           ("lilac", "📓", "Write one thing that went okay today.", 160),
           ("sage", "💧", "Water first. Then the hard problem.", 150),
           ("peach", "🤝", "Asking for help is a skill.", 200)]


@st.cache_resource
def _attempts():
    return {}


def _load_users() -> dict:
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=1), encoding="utf-8")


def hash_pw(pw: str, salt: bytes = None) -> str:
    salt = salt or secrets.token_bytes(16)
    return salt.hex() + "$" + hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 240_000).hex()


def verify_pw(pw: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$")
        return hmac.compare_digest(hash_pw(pw, bytes.fromhex(salt_hex)), stored)
    except (ValueError, TypeError):
        return False


def start_session(username: str, display: str = ""):
    ss.user = username
    ss.persist = username != "guest"
    f = data_file()
    try:
        load_into_state(json.loads(f.read_text(encoding="utf-8")) if f and f.exists() else {})
    except Exception:
        load_into_state({})
    if display and not ss.name:
        ss.name = display
    st.rerun()


def student_svg(skin="#f1c27d", hair="#2b1b12", style="short", glasses=True, phones=False):
    """A friendly student peeking over the edge of a panel, hands on the edge."""
    shade = "rgba(0,0,0,.14)"
    s = ['<svg viewBox="0 0 200 160" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">']
    if style == "pony":
        s.append(f'<ellipse cx="168" cy="98" rx="13" ry="32" fill="{hair}" transform="rotate(-10 168 98)"/>')
    s.append(f'<ellipse cx="42" cy="106" rx="9" ry="13" fill="{skin}"/><ellipse cx="158" cy="106" rx="9" ry="13" fill="{skin}"/>')
    s.append(f'<ellipse cx="100" cy="100" rx="58" ry="60" fill="{skin}"/>')
    s.append(f'<path d="M40 100C34 12 166 12 160 100C152 78 126 66 100 66C74 66 48 78 40 100Z" fill="{hair}"/>')
    if style == "curly":
        s += [f'<circle cx="{cx}" cy="{52 + abs(cx - 100) // 3}" r="17" fill="{hair}"/>' for cx in (54, 72, 90, 110, 128, 146)]
    if phones:
        s.append('<path d="M36 100C34 24 166 24 164 100" fill="none" stroke="#ff6a13" stroke-width="8" stroke-linecap="round"/>'
                 '<rect x="26" y="84" width="18" height="38" rx="9" fill="#ff6a13"/>'
                 '<rect x="156" y="84" width="18" height="38" rx="9" fill="#ff6a13"/>')
    s.append('<circle cx="78" cy="104" r="5.5" fill="#1d1d1d"/><circle cx="122" cy="104" r="5.5" fill="#1d1d1d"/>'
             '<circle cx="76" cy="102" r="1.8" fill="#fff"/><circle cx="120" cy="102" r="1.8" fill="#fff"/>'
             '<path d="M66 88q12-8 24 0M110 88q12-8 24 0" stroke="#1d1d1d" stroke-width="3" fill="none" '
             'stroke-linecap="round" opacity=".7"/>'
             '<circle cx="62" cy="122" r="7" fill="#ff8f8f" opacity=".35"/><circle cx="138" cy="122" r="7" fill="#ff8f8f" opacity=".35"/>'
             '<path d="M86 126q14 14 28 0" stroke="#1d1d1d" stroke-width="3" fill="none" stroke-linecap="round"/>')
    if glasses:
        s.append('<circle cx="78" cy="104" r="17" fill="rgba(255,255,255,.18)" stroke="#1d1d1d" stroke-width="3.5"/>'
                 '<circle cx="122" cy="104" r="17" fill="rgba(255,255,255,.18)" stroke="#1d1d1d" stroke-width="3.5"/>'
                 '<path d="M95 104h10" stroke="#1d1d1d" stroke-width="3.5"/>')
    for x in (34, 130):
        s.append(f'<rect x="{x}" y="128" width="36" height="42" rx="17" fill="{skin}"/>'
                 f'<path d="M{x + 12} 140v10M{x + 24} 140v10" stroke="{shade}" stroke-width="2.5" stroke-linecap="round"/>')
    s.append('</svg>')
    return "".join(s)


def home_art(greeting: str, quote: str) -> str:
    return ('<div class="landing">'
            f'<div class="l-hello">{greeting}</div>'
            '<div class="l-title">Feel steadier.<br>Study smarter.</div>'
            '<div class="l-card l-left"><div class="l-tile">🌬️<span class="l-dot">✦</span></div>'
            '<div class="l-name">Today’s tip: box breathing</div><div class="l-price">60 seconds</div></div>'
            f'<div class="l-card l-right"><div class="l-cap">Today’s thought</div><div class="l-quote">{esc(quote)}</div></div>'
            f'<div class="peek pl">{student_svg("#8d5524", "#1a1a1a", "curly", glasses=False)}</div>'
            f'<div class="peek center">{student_svg("#f1c27d", "#2b1b12", "short", glasses=True, phones=True)}</div>'
            f'<div class="peek pr">{student_svg("#ffdbac", "#6b3a1e", "pony", glasses=True)}</div></div>')


def auth_art() -> str:
    return ('<div class="auth-art"><div class="l-title">Feel steadier.<br>Study smarter.</div>'
            '<p class="auth-sub">Save ideas, plan your focus and breathe, in one calm place made for engineering students.</p>'
            f'<div class="auth-peek">{student_svg("#f1c27d", "#2b1b12", "short", glasses=True, phones=True)}</div>'
            '<div class="auth-base">Free, private and built for students like you.</div></div>')


def theme_hint():
    if st.get_option("theme.base") != "light":
        st.caption("For the full look, put the supplied config.toml in a .streamlit folder next to this app "
                   "and restart it.")


def show_auth():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(auth_art(), unsafe_allow_html=True)
    with right:
        st.title("Welcome to Sage 🌿")
        st.caption("A calm corner for engineering students: save ideas you like, plan your focus, and breathe.")
        tab_in, tab_up, tab_guest = st.tabs(["Sign in", "Create account", "Continue as guest"])
        with tab_in:
            with st.form("signin"):
                u = st.text_input("Username").strip().lower()
                pw = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if go:
                att, users = _attempts(), _load_users()
                n, until = att.get(u, (0, 0.0))
                if n >= 5 and time.time() < until:
                    st.error(f"Too many attempts. Try again in {int(until - time.time())} seconds.")
                elif u in users and verify_pw(pw, users[u]["pw"]):
                    att.pop(u, None)
                    start_session(u, users[u].get("display", ""))
                else:
                    n += 1
                    att[u] = (n, time.time() + 60 if n >= 5 else 0.0)
                    st.error("That username and password don't match. Check both and try again.")
        with tab_up:
            with st.form("signup"):
                disp = st.text_input("What should I call you?", max_chars=30)
                u2 = st.text_input("Choose a username", help="3 to 20 letters, numbers or underscores").strip().lower()
                p1 = st.text_input("Password (8+ characters)", type="password")
                p2 = st.text_input("Confirm password", type="password")
                go2 = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if go2:
                users = _load_users()
                if not re.fullmatch(r"[a-z0-9_]{3,20}", u2):
                    st.error("Usernames need 3 to 20 letters, numbers or underscores.")
                elif u2 in users or u2 == "guest":
                    st.error("That username is taken. Pick another.")
                elif len(p1) < 8:
                    st.error("Passwords need at least 8 characters.")
                elif p1 != p2:
                    st.error("The two passwords don't match.")
                else:
                    users[u2] = {"pw": hash_pw(p1), "display": disp.strip(), "created": dt.date.today().isoformat()}
                    _save_users(users)
                    start_session(u2, disp.strip())
        with tab_guest:
            st.write("Try everything without an account. Nothing is saved when you close the tab.")
            if st.button("Continue as guest", use_container_width=True):
                start_session("guest", "")
    with st.expander("🆘 Need help right now? You don't need to sign in."):
        reg = st.selectbox("Region", list(HELPLINES), key="auth_region")
        for n, num, link, nt in HELPLINES[reg]:
            st.markdown(f"**{n}**: [{num}]({link}) {nt}")
    st.caption("Accounts are stored on this computer only, and passwords are salted and hashed. This is a local "
               "login for a student project, not a cloud service, so don't reuse an important password.")
    theme_hint()


if ss.user is None:
    show_auth()
    st.stop()


# ═════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ═════════════════════════════════════════════════════════════
PAGES = ["🏠 Today", "✨ Discover", "📌 My board", "💬 Talk to Sage", "🍅 Focus", "🔄 Reframe", "🌬️ Breathe", "📈 Insights",
         "🎓 Study tools", "📓 Journal", "🆘 Safety & help"]
with st.sidebar:
    st.markdown("### 🌿 Sage")
    st.markdown(f'<span class="pill">{"Guest" if ss.user == "guest" else "@" + esc(ss.user)}</span>',
                unsafe_allow_html=True)
    page = st.radio("Go to", PAGES, label_visibility="collapsed", key="page")
    if st.button("Sign out", use_container_width=True):
        save_data()
        ss.clear()
        st.rerun()
    st.divider()
    country = st.selectbox("Helpline region", list(HELPLINES), index=0)
    with st.expander("🆘 Need help right now?"):
        for n, num, link, _ in HELPLINES[country][:3]:
            st.markdown(f"[{num}]({link}): {n.split('(')[0].strip()}")
    with st.expander("⚙️ Your data"):
        ss.name = st.text_input("What should I call you?", ss.name)
        ss.persist = st.toggle("Save my data on this computer", ss.persist, disabled=ss.user == "guest",
                               help="Check-ins, goals, journal and safety plan are saved in a sage_data folder "
                                    "next to this app. Chats are never saved.")
        st.download_button("Download backup (JSON)", json.dumps({k: ss[k] for k in DEFAULTS}, indent=1),
                           "sage_backup.json", use_container_width=True)
        up = st.file_uploader("Restore a backup", type="json")
        if up is not None and st.button("Restore now", use_container_width=True):
            try:
                load_into_state(json.load(up))
                st.success("Backup restored.")
            except Exception:
                st.error("That file isn't a valid Sage backup.")
        if st.button("Erase all my data", use_container_width=True):
            load_into_state({})
            ss.messages = ss.messages[:1]
            f = data_file()
            if f and f.exists():
                f.unlink()
            st.rerun()
        if ss.user != "guest":
            sure = st.checkbox("I understand deleting my account can't be undone", key="del_ok")
            if st.button("Delete my account", disabled=not sure, use_container_width=True):
                users = _load_users()
                users.pop(ss.user, None)
                _save_users(users)
                f = data_file()
                if f and f.exists():
                    f.unlink()
                ss.clear()
                st.rerun()
    st.caption("Sage is a self-help companion, not a doctor or therapist.")

if ss.crisis and page != PAGES[-1]:
    render_helplines(country)
    if st.button("I'm safe for now, hide this"):
        ss.crisis = False
        st.rerun()


# ═════════════════════════════════════════════════════════════
# PAGE: TODAY
# ═════════════════════════════════════════════════════════════
AFFIRMATIONS = [
    "You don't need to finish everything today. You need to take the next small step.",
    "A low mark is feedback about one exam, not a verdict on you.",
    "Rest is part of studying. Your brain files things while you sleep.",
    "Asking for help is what strong engineers do. Nobody debugs alone.",
    "Progress counts even when it's slow, messy or invisible.",
    "Comparing your chapter 3 with someone else's chapter 20 isn't fair to you.",
    "You have survived every hard day so far. That's a 100% record.",
]


def page_today():
    st.markdown(WIDE, unsafe_allow_html=True)
    h = dt.datetime.now().hour
    hello = "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"
    goals = todays_goals()
    done, total = sum(goals.values()), len(goals)
    who = f", {esc(ss.name)}" if ss.name else ""
    st.markdown(home_art(f"{hello}{who}", AFFIRMATIONS[dt.date.today().toordinal() % len(AFFIRMATIONS)]),
                unsafe_allow_html=True)

    with st.container(key="band"):
        a, b, c = st.columns(3, gap="small")
        with a:
            with st.container(key="pL"):
                st.markdown(f'<div class="stat"><div class="row"><div class="big">{streak()}</div>'
                            '<div class="avs"><span>🙂</span><span>😌</span><span>💪</span></div></div>'
                            '<div class="sub">day streak. Small, steady check-ins keep it going.</div></div>',
                            unsafe_allow_html=True)
        with b:
            with st.container(key="pC"):
                st.markdown('<div class="cta-title">Take the next small step</div>', unsafe_allow_html=True)
                x1, x2 = st.columns(2)
                x1.button("Start focus →", type="primary", on_click=goto, args=("Focus",),
                          use_container_width=True, key="h_focus")
                x2.button("Breathe", on_click=goto, args=("Breathe",), use_container_width=True, key="h_breathe")
        with c:
            with st.container(key="pR"):
                st.markdown(f'<div class="stat"><div class="big">{done}/{total}<span class="star">★</span></div>'
                            '<div class="sub">goals done today: check-in, focus block, water and one small win.</div></div>',
                            unsafe_allow_html=True)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.container(key="checkin"):
            st.subheader("Quick check-in")
            faces = {"😞": 2, "😕": 4, "😐": 6, "🙂": 8, "😄": 10}
            face = st.select_slider("How are you feeling?", list(faces), value="😐")
            c1, c2 = st.columns(2)
            stress = c1.slider("Stress", 1, 10, 5)
            sleep = c2.number_input("Sleep last night (hrs)", 0.0, 14.0, 7.0, 0.5)
            txt = st.text_input("One line about today (optional)")
            if st.button("Save check-in", type="primary"):
                if not guard(txt, country):
                    ss.moods.append({"date": dt.datetime.now().isoformat(timespec="minutes"), "mood": faces[face],
                                     "stress": stress, "sleep": sleep, "note": txt})
                    st.success("Saved. Thanks for checking in with yourself.")
            last = ss.moods[-1] if ss.moods else None
            if last and last["date"][:10] == TODAY:
                if last["stress"] >= 8:
                    note("Your stress is high. Try two minutes on the **Breathe** page before you start studying.", lamp=True)
                elif last["sleep"] < 5:
                    note("You slept very little. Plan lighter, shorter focus blocks today and go to bed early tonight.", lamp=True)
                elif ss.tasks:
                    note("You have tasks waiting. Open **Focus** and start just one 25-minute block.")
    with right:
        with st.container(key="water"):
            st.subheader("Water")
            w = ss.water.get(TODAY, 0)
            dots = "".join(f'<span class="gl{" on" if i < w else ""}"></span>' for i in range(8))
            st.markdown(f'<div class="glasses">{dots}</div>', unsafe_allow_html=True)
            st.caption(f"{w} of 8 glasses today")
            c1, c2 = st.columns(2)
            if c1.button("＋ glass", use_container_width=True):
                ss.water[TODAY] = w + 1
                st.rerun()
            if c2.button("− glass", use_container_width=True) and w > 0:
                ss.water[TODAY] = w - 1
                st.rerun()
        due = sum(1 for c in ss.srs if c["next_due"] <= TODAY)
        if due:
            note(f"🧠 **{due}** topic{'s' if due != 1 else ''} due for review in **Study tools**.")


# ═════════════════════════════════════════════════════════════
# PAGE: TALK
# ═════════════════════════════════════════════════════════════
def page_talk():
    st.title("Talk to Sage")
    mode = st.radio("What would help most?", list(MODES), horizontal=True)
    extra, starters = MODES[mode]
    for m in ss.messages:
        with st.chat_message(m["role"], avatar="🌿" if m["role"] == "assistant" else None):
            st.markdown(m["content"])

    def send(text):
        ss.messages.append({"role": "user", "content": text})
        if crisis_detected(text):
            ss.crisis = True
            ss.messages.append({"role": "assistant", "content": CRISIS_CHAT_MESSAGE, "safety": True})
            return
        reply = ask_model(ss.messages, extra)
        if crisis_detected(reply):
            ss.crisis = True
            ss.messages.append({"role": "assistant", "content": CRISIS_CHAT_MESSAGE, "safety": True})
        else:
            ss.messages.append({"role": "assistant", "content": reply})

    if len(ss.messages) <= 2:
        st.caption("Not sure where to start? Tap one:")
        cols = st.columns(len(starters))
        for col, s in zip(cols, starters):
            if col.button(s, use_container_width=True, key="st_" + s):
                send(s)
                st.rerun()
    p = st.chat_input("Tell Sage what's on your mind…")
    if p:
        send(p)
        st.rerun()
    if len(ss.messages) > 1:
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in ss.messages)
        c1, c2 = st.columns(2)
        c1.download_button("Download this chat", transcript, "sage_chat.txt")
        if c2.button("Start a new chat"):
            ss.messages = ss.messages[:1]
            st.rerun()


# ═════════════════════════════════════════════════════════════
# PAGE: FOCUS (planner, timer, sounds)
# ═════════════════════════════════════════════════════════════
def build_schedule(tasks, start, focus=25, short=5, long_=15, every=4):
    rows, t, count = [], start, 0
    for task in sorted(tasks, key=lambda x: (x["priority"], x["due"])):
        blocks = max(1, math.ceil(task["minutes"] / focus))
        for b in range(blocks):
            rows.append({"Start": t.strftime("%H:%M"), "End": (t + dt.timedelta(minutes=focus)).strftime("%H:%M"),
                         "Type": "🍅 Focus", "Task": f'{task["name"]} ({b + 1}/{blocks})'})
            t += dt.timedelta(minutes=focus)
            count += 1
            brk = long_ if count % every == 0 else short
            rows.append({"Start": t.strftime("%H:%M"), "End": (t + dt.timedelta(minutes=brk)).strftime("%H:%M"),
                         "Type": "☕ Long break" if brk == long_ else "☕ Break",
                         "Task": "Stand up, drink water, look far away"})
            t += dt.timedelta(minutes=brk)
    return rows


TIMER_HTML = """
<div style="font-family:'Plus Jakarta Sans',sans-serif;text-align:center;padding:16px;border-radius:18px;background:#0a4a1a;color:#eaf6f1">
 <div id="mode" style="font-size:15px;color:#ff9d42">Focus</div>
 <div id="t" style="font-size:60px;font-weight:700;font-family:'Plus Jakarta Sans',sans-serif">__F__:00</div>
 <button onclick="go()" style="padding:9px 20px;border-radius:10px;border:0;background:#ff9d42;color:#0a4a1a;font-weight:700;cursor:pointer">Start</button>
 <button onclick="pa()" style="padding:9px 20px;border-radius:10px;border:0;background:#ff6a13;color:#10282c;font-weight:700;cursor:pointer">Pause</button>
 <button onclick="re()" style="padding:9px 20px;border-radius:10px;border:0;background:#c9d8d2;color:#10282c;font-weight:700;cursor:pointer">Reset</button>
 <div id="n" style="margin-top:8px;font-size:13px;color:#d9f7d0">Blocks completed: 0 (log them below)</div>
</div>
<script>
 let F=__F__*60,B=__B__*60,left=F,focus=true,iv=null,done=0;
 const t=document.getElementById('t'),m=document.getElementById('mode'),n=document.getElementById('n');
 function draw(){t.textContent=String(Math.floor(left/60)).padStart(2,'0')+':'+String(left%60).padStart(2,'0');}
 function beep(){try{const c=new AudioContext(),o=c.createOscillator();o.connect(c.destination);o.frequency.value=660;o.start();setTimeout(()=>o.stop(),450);}catch(e){}}
 function tick(){left--;if(left<=0){beep();if(focus){done++;n.textContent='Blocks completed: '+done+' (log them below)';}focus=!focus;left=focus?F:B;m.textContent=focus?'Focus':'Break: stretch and sip water';}draw();}
 function go(){if(!iv)iv=setInterval(tick,1000);}
 function pa(){clearInterval(iv);iv=null;}
 function re(){pa();focus=true;left=F;m.textContent='Focus';draw();}
</script>"""

SOUNDS_HTML = """
<div style="font-family:'Plus Jakarta Sans',sans-serif;padding:12px 14px;border-radius:14px;background:#ff9d4222;border:1px solid #ff9d4266">
 <b>Focus sounds</b> <span style="font-size:13px;opacity:.8">(generated on your device, no internet needed)</span><br><br>
 <button onclick="go('brown')">Brown noise</button> <button onclick="go('rain')">Rain-like</button>
 <button onclick="go('white')">White noise</button> <button onclick="stop()">Stop</button>
 &nbsp; Volume <input id="v" type="range" min="5" max="100" value="35" oninput="vol()">
</div>
<script>
 let ctx,src,g;
 function stop(){if(src){try{src.stop()}catch(e){}src=null;}}
 function vol(){if(g)g.gain.value=document.getElementById('v').value/100*0.6;}
 function go(type){
  if(!ctx)ctx=new (window.AudioContext||window.webkitAudioContext)(); stop();
  const len=ctx.sampleRate*3,buf=ctx.createBuffer(1,len,ctx.sampleRate),d=buf.getChannelData(0);let last=0;
  for(let i=0;i<len;i++){const w=Math.random()*2-1; if(type==='brown'){last=(last+0.02*w)/1.02;d[i]=last*3.5;} else d[i]=w*0.5;}
  src=ctx.createBufferSource();src.buffer=buf;src.loop=true;
  const f=ctx.createBiquadFilter();f.type=type==='rain'?'bandpass':(type==='brown'?'lowpass':'allpass');
  f.frequency.value=type==='rain'?1800:800; g=ctx.createGain(); vol();
  src.connect(f);f.connect(g);g.connect(ctx.destination);src.start();
 }
</script>"""


def page_focus():
    st.title("Focus")
    st.caption("List what's on your plate. Sage splits it into focus blocks with real breaks.")
    with st.form("task_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.4])
        name = c1.text_input("Task", placeholder="e.g. DAA lab record, Exp 5")
        mins = c2.number_input("Minutes", 10, 600, 50, step=5)
        prio = c3.selectbox("Priority", [1, 2, 3], format_func=lambda x: {1: "High", 2: "Medium", 3: "Low"}[x])
        due = c4.date_input("Due", dt.date.today())
        if st.form_submit_button("Add task") and name.strip():
            if not guard(name, country):
                ss.tasks.append({"name": name.strip(), "minutes": int(mins), "priority": prio, "due": due.isoformat()})
    if not ss.tasks:
        note("No tasks yet. Add one above. Even a vague one like \"start lab record\" is fine.")
    else:
        st.dataframe(pd.DataFrame(ss.tasks), hide_index=True, use_container_width=True)
        finished = st.multiselect("Finished any of these?", [t["name"] for t in ss.tasks])
        if finished and st.button("Mark finished 🎉"):
            ss.tasks = [t for t in ss.tasks if t["name"] not in finished]
            for f in finished:
                ss.wins.append({"date": TODAY, "text": f"Finished: {f}"})
            st.rerun()
        c1, c2, c3 = st.columns(3)
        fl = c1.slider("Focus (min)", 15, 50, 25, 5)
        bl = c2.slider("Break (min)", 3, 15, 5)
        stt = c3.time_input("Start at", dt.datetime.now().time().replace(second=0, microsecond=0))
        if st.button("Build my plan", type="primary"):
            sched = build_schedule(ss.tasks, dt.datetime.combine(dt.date.today(), stt), fl, bl)
            st.dataframe(pd.DataFrame(sched), hide_index=True, use_container_width=True)
            n = sum("Focus" in r["Type"] for r in sched)
            note(f"{n} focus blocks. You don't have to do them all today. Start with the first one.")
            st.download_button("Download plan (CSV)", pd.DataFrame(sched).to_csv(index=False), "pomodoro_plan.csv")
    st.divider()
    st.subheader("Timer")
    c1, c2 = st.columns(2)
    f = c1.number_input("Focus minutes", 5, 90, 25, key="tf")
    b = c2.number_input("Break minutes", 1, 30, 5, key="tb")
    components.html(TIMER_HTML.replace("__F__", str(int(f))).replace("__B__", str(int(b))), height=235)
    c1, c2 = st.columns([1, 2])
    if c1.button("✅ I finished a focus block"):
        ss.focus_log.append(TODAY)
        st.rerun()
    c2.markdown(f"Today: **{ss.focus_log.count(TODAY)}** blocks. All time: **{len(ss.focus_log)}**")
    components.html(SOUNDS_HTML, height=140)


# ═════════════════════════════════════════════════════════════
# PAGE: REFRAME
# ═════════════════════════════════════════════════════════════
DISTORTIONS = {
    "All-or-nothing": ["always", "never", "completely", "totally", "perfect", "everything", "nothing"],
    "Catastrophising": ["ruin", "disaster", "end of", "worst", "no future", "doomed", "fail everything"],
    "Mind-reading": ["they think", "everyone thinks", "people think", "they'll judge", "must think"],
    "Should-statements": ["should", "must", "have to", "ought to"],
    "Labelling": ["i'm stupid", "i am stupid", "i'm dumb", "i am dumb", "i'm a failure", "i am a failure", "useless", "idiot", "loser"],
    "Fortune-telling": ["will fail", "will never", "won't be able", "going to fail", "can't do"],
    "Comparing": ["smarter than me", "better than me", "everyone else", "behind everyone"],
}


def page_reframe():
    st.title("Reframe a thought")
    st.caption("A thought is not a fact. Let's test this one.")
    thought = st.text_area("What's the thought that's bothering you?",
                           placeholder="e.g. I'll never understand algorithms. I'm going to fail the semester.")
    if not thought:
        note("Write the thought exactly as it sounds in your head. We'll work with it from there.")
        return
    if guard(thought, country):
        return
    found = [d for d, ks in DISTORTIONS.items() if any(k in thought.lower() for k in ks)]
    if found:
        note("Possible thinking traps: **" + ", ".join(found) + "**. They're common and very human.")
    c1, c2 = st.columns(2)
    ev_for = c1.text_area("Evidence for the thought")
    ev_against = c2.text_area("Evidence against it (past wins, help available, partial progress)")
    friend = st.text_input("What would you tell a friend who thought this?")
    balanced = st.text_area("A fairer, more balanced thought")
    c1, c2 = st.columns(2)
    before = c1.slider("Belief in the original thought (%)", 0, 100, 80)
    after = c2.slider("Belief after reframing (%)", 0, 100, 50)
    b1, b2 = st.columns(2)
    if b1.button("Save to journal") and balanced.strip():
        if not guard(balanced, country):
            ss.reframes.append({"date": dt.datetime.now().strftime("%d %b %H:%M"), "thought": thought,
                                "balanced": balanced, "before": before, "after": after})
            st.success(f"Saved. Your belief dropped by {max(before - after, 0)} points.")
    if b2.button("✨ Ask Sage to help"):
        if any(crisis_detected(x) for x in (ev_for, ev_against, friend, balanced)):
            ss.crisis = True
            render_helplines(country)
        else:
            with st.spinner("Thinking gently…"):
                out = ask_model([{"role": "user", "content":
                                  f"Help me reframe this thought using CBT. Thought: {thought}. Evidence for: "
                                  f"{ev_for or 'n/a'}. Evidence against: {ev_against or 'n/a'}. Name the likely "
                                  "distortion, then give 2 balanced alternative thoughts. Under 110 words."}])
            if crisis_detected(out):
                ss.crisis = True
                render_helplines(country)
            else:
                st.markdown(out)
    if ss.reframes:
        with st.expander(f"My reframing journal ({len(ss.reframes)})"):
            st.dataframe(pd.DataFrame(ss.reframes), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE: BREATHE
# ═════════════════════════════════════════════════════════════
PATTERNS = {"Box breathing (4-4-4-4)": (4, 4, 4, 4), "4-7-8 relaxing": (4, 7, 8, 0),
            "Calm exhale (4-0-6-0)": (4, 0, 6, 0)}

BREATH_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;font-family:'Plus Jakarta Sans',sans-serif;padding:10px">
<style>
 @keyframes breathe{0%{transform:scale(.55)}__A__%{transform:scale(1)}__B__%{transform:scale(1)}__C__%{transform:scale(.55)}100%{transform:scale(.55)}}
 .orb{width:200px;height:200px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#e6ffe0,#3fae5a);
  animation:breathe __T__s ease-in-out __R__;box-shadow:0 0 44px #9fe2a4;}
 @media (prefers-reduced-motion:reduce){.orb{animation:none}}
</style>
<div class="orb"></div>
<div id="lbl" style="margin-top:16px;font-size:24px;height:32px;font-family:'Plus Jakarta Sans',sans-serif;color:#0a4a1a"></div>
<div id="cnt" style="font-size:14px;opacity:.7"></div></div>
<script>
 const seq=[["Breathe in",__I__],["Hold",__H1__],["Breathe out",__E__],["Hold",__H2__]].filter(s=>s[1]>0);
 let r=0,i=0,left=seq[0][1];const l=document.getElementById('lbl'),c=document.getElementById('cnt');
 function show(){l.textContent=seq[i][0]+'  '+left;c.textContent='Round '+(r+1)+' of __R__';}
 show();
 const iv=setInterval(()=>{left--;if(left<=0){i++;if(i>=seq.length){i=0;r++;}
  if(r>=__R__){clearInterval(iv);l.textContent='Well done';c.textContent='Notice how your body feels now.';return;}
  left=seq[i][1];}show();},1000);
</script>"""


def page_breathe():
    st.title("Breathe")
    pname = st.selectbox("Pattern", list(PATTERNS))
    rounds = st.slider("Rounds", 3, 10, 5)
    inh, h1, exh, h2 = PATTERNS[pname]
    total = inh + h1 + exh + h2
    pc = lambda s: f"{s / total * 100:.1f}"
    html = (BREATH_HTML.replace("__A__", pc(inh)).replace("__B__", pc(inh + h1)).replace("__C__", pc(inh + h1 + exh))
            .replace("__T__", str(total)).replace("__R__", str(rounds)).replace("__I__", str(inh))
            .replace("__H1__", str(h1)).replace("__E__", str(exh)).replace("__H2__", str(h2)))
    components.html(html, height=340)
    st.caption("Breathe through your nose if you can. If you feel dizzy, go back to normal breathing.")
    with st.expander("5-4-3-2-1 grounding, for a racing mind"):
        st.markdown("Name **5** things you see, **4** you can touch, **3** you hear, **2** you smell, **1** you taste.")
    with st.expander("Progressive muscle release (3 minutes)"):
        st.markdown("Tense each area for 5 seconds, then release for 10: **hands, arms, shoulders, face, stomach, "
                    "legs, feet.** Notice the difference between tense and relaxed.")


# ═════════════════════════════════════════════════════════════
# PAGE: INSIGHTS (+ burnout check)
# ═════════════════════════════════════════════════════════════
BURNOUT_Q = [
    "I feel drained even after resting",
    "I'm avoiding or putting off my work more than usual",
    "I find it hard to concentrate or remember things",
    "I feel irritable or short-tempered with people",
    "I've lost interest in things I used to enjoy",
    "I feel hopeless about my future",
]


def page_insights():
    st.title("Insights")
    if len(ss.moods) < 2:
        note("Log a few daily check-ins on the **Today** page and your patterns will show up here.")
    else:
        df = pd.DataFrame(ss.moods)
        df["when"] = pd.to_datetime(df["date"])
        wk = df[df["when"] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Check-ins (7d)", len(wk))
        c2.metric("Avg mood", f"{wk['mood'].mean():.1f}" if len(wk) else "–")
        c3.metric("Avg stress", f"{wk['stress'].mean():.1f}" if len(wk) else "–")
        c4.metric("Avg sleep", f"{wk['sleep'].mean():.1f} h" if len(wk) else "–")
        st.line_chart(df.set_index("when")[["mood", "stress"]])
        if len(df) >= 5 and df["sleep"].std() > 0 and df["mood"].std() > 0:
            r = df["sleep"].corr(df["mood"])
            if r >= 0.3:
                note(f"On days you sleep more, your mood tends to be higher (correlation {r:.2f}). Sleep may be your "
                     "strongest lever.", lamp=True)
        if len(df) >= 3 and df["mood"].tail(3).mean() <= 4:
            note("Your mood has been low across your last few check-ins. Consider talking to a friend, your "
                 "college counsellor or a helpline (see **Safety & help**).", lamp=True)
        st.download_button("Export check-ins (CSV)", df.drop(columns="when").to_csv(index=False), "mood_log.csv")

    st.divider()
    st.subheader("Burnout self-check")
    st.caption("A quick reflection, not a diagnosis. In the last two weeks, how often has each been true?")
    labels = ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
    with st.form("burnout"):
        ans = [labels.index(st.select_slider(q, labels, value="Rarely", key=f"bq{i}")) for i, q in enumerate(BURNOUT_Q)]
        go = st.form_submit_button("See my result", type="primary")
    if go:
        total = sum(ans)
        if ans[5] >= 3 or ans[4] >= 4:
            note("Some of your answers suggest you're carrying something heavy. Please consider talking to a "
                 "counsellor or someone you trust soon.", lamp=True)
        if ans[5] >= 3:
            render_helplines(country)
        if total <= 8:
            st.success(f"Score {total}/24: low signs of burnout. Keep protecting your sleep, food and breaks.")
        elif total <= 15:
            st.warning(f"Score {total}/24: moderate signs. Lighten your load where you can, schedule real breaks, "
                       "and talk to someone this week.")
        else:
            st.error(f"Score {total}/24: high signs of burnout. Please speak with your college counsellor, mentor or "
                     "a doctor soon, and cut non-essential commitments for now.")


# ═════════════════════════════════════════════════════════════
# PAGE: STUDY TOOLS
# ═════════════════════════════════════════════════════════════
SOS = {
    "Night before the exam": ["Stop new topics after 8pm. Skim your own notes and formula sheet only.",
                              "Pack your hall ticket, ID, pens and calculator now.",
                              "Eat a proper dinner, drink water, put the phone away by 10:30pm.",
                              "Sleep beats a last hour of cramming. Set two alarms."],
    "Panic inside the exam hall": ["Breathe out slowly for 6 counts, three times.",
                                   "Write down everything you remember on the rough sheet first.",
                                   "Start with the question you know best to build momentum.",
                                   "Blank on one question? Skip it. Your brain will keep working on it."],
    "Viva or interview nerves": ["Slow your speech by 20%. Pauses look confident.",
                                 "It's fine to say \"Let me think for a moment.\"",
                                 "If you don't know: say what you do know, then how you'd find out.",
                                 "Prepare a 30-second intro and two project stories."],
    "Presentation fear": ["Rehearse out loud twice, standing up.",
                          "Open with one question or one number, not an apology.",
                          "Look at one friendly face at a time.",
                          "Shaky hands? Hold a pen or rest them on the podium."],
    "I can't even start": ["Pick the smallest piece: open the file, write the title.",
                           "Set a 5-minute timer and promise yourself you can stop after.",
                           "Remove one distraction: phone in another room.",
                           "Messy first, good later. Done is better than perfect."],
}
TECHNIQUES = {
    "Active recall": "Close the book and write or explain what you remember. Then check. Far better than rereading.",
    "Spaced repetition": "Review after 1, 3, 7, 14 and 30 days. Use the Spaced review tab here.",
    "Feynman technique": "Explain the topic as if teaching a first-year. Gaps in your explanation are gaps in your understanding.",
    "Past papers": "Do at least 3 previous papers under time limits. Mark them honestly and note patterns.",
    "Interleaving": "Mix problem types in one session (e.g. sorting, then DP, then graphs) instead of one topic for hours.",
}


def page_study():
    st.title("Study tools")
    t1, t2, t3, t4, t5 = st.tabs(["Spaced review", "Exam countdown", "Priorities", "Calm kits", "Techniques"])
    with t1:
        st.caption("Add a topic after you study it. Sage tells you when to revise it so it sticks.")
        c1, c2 = st.columns([4, 1])
        topic = c1.text_input("Topic", placeholder="e.g. Dijkstra's algorithm", label_visibility="collapsed")
        if c2.button("Add topic") and topic.strip():
            if not guard(topic, country):
                ss.srs.append({"topic": topic.strip(), "stage": 0,
                               "next_due": (dt.date.today() + dt.timedelta(days=1)).isoformat()})
                st.rerun()

        def advance(i, ok):
            c = ss.srs[i]
            if ok:
                c["stage"] += 1
                if c["stage"] >= len(INTERVALS):
                    ss.srs.pop(i)
                    ss.wins.append({"date": TODAY, "text": f"Mastered: {c['topic']}"})
                    return
            else:
                c["stage"] = 0
            c["next_due"] = (dt.date.today() + dt.timedelta(days=INTERVALS[c["stage"]])).isoformat()

        due = [(i, c) for i, c in enumerate(ss.srs) if c["next_due"] <= TODAY]
        if due:
            st.markdown(f"**{len(due)} to review today**")
        for i, c in due:
            a, b, d = st.columns([4, 1.2, 1.2])
            a.write(c["topic"])
            b.button("Got it", key=f"ok{i}", on_click=advance, args=(i, True))
            d.button("Forgot", key=f"no{i}", on_click=advance, args=(i, False))
        if ss.srs and not due:
            note("Nothing due today. Next review: " + min(c["next_due"] for c in ss.srs))
        if not ss.srs:
            note("No topics yet. Add the first one you studied today.")
    with t2:
        en = st.text_input("Exam or submission", "Semester exam")
        ed = st.date_input("Date", dt.date.today() + dt.timedelta(days=21), key="exam")
        topics = st.number_input("Topics left", 1, 200, 12)
        left = (ed - dt.date.today()).days
        if left > 0:
            per = math.ceil(topics / max(left - 2, 1))
            c1, c2 = st.columns(2)
            c1.metric(f"Days until {en}", left)
            c2.metric("Topics per day", per)
            note("This keeps the last 2 days free for revision and rest. Small and steady beats a last-night rush.")
        else:
            st.warning("Pick a date in the future.")
    with t3:
        st.caption("One task per line. Add ! if urgent and * if important, e.g. `Lab record !*`")
        items = st.text_area("Tasks", height=130, label_visibility="collapsed")
        if items.strip():
            q = {"Do now": [], "Schedule it": [], "Quick or delegate": [], "Drop or later": []}
            for line in filter(None, (x.strip() for x in items.splitlines())):
                u, im = "!" in line, "*" in line
                k = list(q)[0] if u and im else list(q)[1] if im else list(q)[2] if u else list(q)[3]
                q[k].append(line.replace("!", "").replace("*", "").strip())
            cols = st.columns(2)
            for n, (k, v) in enumerate(q.items()):
                cols[n % 2].markdown(f"**{k}**\n" + ("\n".join(f"- {x}" for x in v) or "- none"))
    with t4:
        kit = st.selectbox("What are you facing?", list(SOS))
        for step in SOS[kit]:
            st.checkbox(step, key=f"{kit}{step}")
    with t5:
        for k, v in TECHNIQUES.items():
            st.markdown(f"**{k}.** {v}")


# ═════════════════════════════════════════════════════════════
# PAGE: JOURNAL
# ═════════════════════════════════════════════════════════════
def page_journal():
    st.title("Journal")
    t1, t2 = st.tabs(["Gratitude", "Small wins"])
    with t1:
        g = st.text_area("One thing that went okay today, however small")
        if st.button("Save entry") and g.strip():
            if not guard(g, country):
                ss.gratitude.append({"date": TODAY, "text": g.strip()})
                st.success("Saved.")
        for e in reversed(ss.gratitude[-6:]):
            st.markdown(f"**{e['date']}**: {e['text']}")
        if not ss.gratitude:
            note("Your entries will collect here. On a hard day, reading them back helps.")
    with t2:
        w = st.text_input("Something you did today")
        if st.button("Add win") and w.strip():
            if not guard(w, country):
                ss.wins.append({"date": TODAY, "text": w.strip()})
        for e in reversed(ss.wins[-10:]):
            st.markdown(f"🏆 **{e['date']}**: {e['text']}")
        if not ss.wins:
            note("Finished a lab? Understood a topic? Got out of bed on a hard day? It all counts.")


# ═════════════════════════════════════════════════════════════
# PAGE: SAFETY & HELP
# ═════════════════════════════════════════════════════════════
def page_safety():
    st.title("Safety & help")
    if ss.crisis:
        render_helplines(country)
        if st.button("I'm safe for now, hide the alert"):
            ss.crisis = False
            st.rerun()
    else:
        st.markdown("#### 📞 Free helplines")
        for n, num, link, nt in HELPLINES[country]:
            st.markdown(f"**{n}**: [{num}]({link}) {nt}")
    st.caption("Numbers can change. Please verify on the official Tele-MANAS, KIRAN and iCall sites. "
               "Your college counselling cell and mentor are good first contacts too.")
    st.divider()
    st.subheader("My safety plan")
    st.caption("Fill this in on a calm day, so it's ready on a hard one. It stays on your device.")
    sp = ss.safety_plan
    fields = [("signs", "My warning signs (thoughts, moods, situations)"),
              ("alone", "Things I can do on my own to feel safer"),
              ("places", "People or places that help me feel better"),
              ("people", "People I can ask for help (names and numbers)"),
              ("counsellor", "My college counsellor or mentor (name and contact)"),
              ("safer", "How I can make my surroundings safer"),
              ("reason", "What matters most to me")]
    for k, label in fields:
        sp[k] = st.text_area(label, sp.get(k, ""), height=70)
    if any(sp.values()):
        text = "MY SAFETY PLAN\n\n" + "\n\n".join(f"{lbl}\n{sp.get(k, '')}" for k, lbl in fields)
        text += "\n\nHelplines: " + "; ".join(f"{n}: {num}" for n, num, _, _ in HELPLINES[country])
        st.download_button("Download my plan", text, "my_safety_plan.txt")


# ═════════════════════════════════════════════════════════════
# PAGES: DISCOVER (pin feed) and MY BOARD
# ═════════════════════════════════════════════════════════════
PAGE_BY_KEY = {p.split(" ", 1)[1]: p for p in PAGES}

# (id, category, emoji, title, body, tint, height, "Try it" target page)
_RAW = [
    ("p01", "Calm", "🌬️", "Box breathing in 60 seconds", "Breathe in for 4, hold 4, out 4, hold 4. Four rounds is enough to settle your shoulders.", "sky", 230, "Breathe"),
    ("p02", "Calm", "🖐️", "5-4-3-2-1 grounding", "When your mind is racing, name what you can see, touch, hear, smell and taste.", "sage", 210, "Breathe"),
    ("p03", "Calm", "🌙", "", "Breathe out slowly. You've handled every hard day so far.", "lilac", 190, None),
    ("p04", "Calm", "🚶", "Five minutes outside", "Step out, look at something far away, and let your eyes rest from the screen.", "butter", 170, None),
    ("p05", "Calm", "😌", "Unclench your jaw", "Drop your shoulders, relax your jaw, open your hands. Most of us hold stress there.", "blush", 160, None),
    ("p06", "Focus", "🍅", "The 25-minute deal", "Work for 25 minutes on one thing, then take a real 5-minute break. Repeat.", "blush", 220, "Focus"),
    ("p07", "Focus", "📵", "Phone in another room", "Out of sight is out of mind. Your focus blocks get noticeably better.", "lilac", 170, None),
    ("p08", "Focus", "🐸", "Eat the frog first", "Do the task you're dreading while your energy is fresh. Everything after feels lighter.", "sage", 200, None),
    ("p09", "Focus", "⏱️", "The two-minute start", "Can't begin? Open the file and write the title. Starting is the hardest part.", "peach", 210, "Focus"),
    ("p10", "Focus", "🎧", "Try focus sounds", "Brown noise and rain-like sounds mask distractions without lyrics. Find them on the Focus page.", "sky", 180, "Focus"),
    ("p11", "Study", "🧠", "Active recall beats rereading", "Close the book and explain it from memory. Then check what you missed.", "sky", 230, "Study tools"),
    ("p12", "Study", "📅", "Spaced review", "Revise a topic after 1, 3, 7, 14 and 30 days so it actually stays.", "butter", 200, "Study tools"),
    ("p13", "Study", "🗣️", "Teach it to a first-year", "If you can explain Dijkstra simply, you understand it. Gaps show up fast.", "peach", 190, None),
    ("p14", "Study", "📄", "Do three past papers", "Time yourself and mark honestly. Patterns in questions show up quickly.", "lilac", 180, None),
    ("p15", "Study", "🌙", "Night before the exam", "Skim only your own notes, pack your bag, eat well and sleep.", "blush", 210, "Study tools"),
    ("p16", "Self-care", "💧", "Water first", "A glass of water before the hard problem. Tiredness is often thirst.", "sky", 150, "Today"),
    ("p17", "Self-care", "😴", "Sleep beats cramming", "Memory is stored while you sleep. A full night is part of the revision.", "lilac", 200, None),
    ("p18", "Self-care", "🍛", "Eat a proper meal", "Skipping lunch to study costs you focus later. Fuel is part of the plan.", "peach", 170, None),
    ("p19", "Self-care", "🤸", "Move for ten minutes", "A short walk or stretch resets your mood and your attention.", "sage", 190, None),
    ("p20", "Self-care", "💬", "Talk to one person today", "A friend, a mentor, a family member. You're not meant to carry it alone.", "butter", 220, None),
    ("p21", "Motivation", "🌱", "Small wins count", "Finished a lab? Understood one topic? Write it down. Wins add up.", "sage", 200, "Journal"),
    ("p22", "Motivation", "💛", "", "A low mark is feedback about one exam, not a verdict on you.", "blush", 230, None),
    ("p23", "Motivation", "🔄", "Test the thought", "\"I'll never get this\" is a thought, not a fact. Write down the evidence for and against.", "butter", 220, "Reframe"),
    ("p24", "Motivation", "🤝", "Asking for help is a skill", "Nobody debugs alone. Ask a senior, a teacher or your college counsellor.", "lilac", 190, None),
    ("p25", "Motivation", "🌿", "", "Progress counts even when it's slow, messy or invisible.", "sky", 200, None),
    ("p26", "Motivation", "📌", "Rejection is data", "A placement rejection says the fit wasn't right this time. Adjust, practise, try again.", "peach", 210, None),
    ("p27", "Motivation", "🧭", "A backlog is a detour", "It's a setback, not the end of your degree. Plan one subject at a time.", "sky", 200, "Focus"),
    ("p28", "Motivation", "🪞", "", "Comparing your chapter 3 with someone else's chapter 20 isn't fair to you.", "butter", 220, None),
]
STEPS = {
    "p01": ["Sit comfortably and relax your shoulders.", "Breathe in through your nose for 4 seconds.",
            "Hold for 4. Breathe out slowly for 4. Hold for 4.", "Repeat 4 times, then notice how you feel."],
    "p02": ["Name 5 things you can see.", "Touch 4 things around you.", "Listen for 3 sounds.",
            "Notice 2 smells.", "Notice 1 taste."],
    "p06": ["Pick one task.", "Set a 25-minute timer and silence your phone.", "Work only on that task.",
            "Take a 5-minute break away from the screen.", "After 4 rounds, take a 15-minute break."],
    "p09": ["Open the file or book.", "Write the title or the first line.", "Keep going for 2 minutes.",
            "You may stop after that, but most people don't."],
    "p11": ["Close the book.", "Write or say everything you remember about the topic.",
            "Open the book and mark what you missed.", "Revise the missed parts tomorrow."],
    "p15": ["Stop new topics after 8pm.", "Skim your own notes and formula sheet.",
            "Pack your hall ticket, ID, pens and calculator.", "Eat dinner, drink water, phone away by 10:30pm.",
            "Sleep, and set two alarms."],
}
PINS = [dict(id=i, cat=c, emoji=e, title=t, body=b, tint=tn, h=h, action=a, steps=STEPS.get(i))
        for (i, c, e, t, b, tn, h, a) in _RAW]
PIN_BY_ID = {p["id"]: p for p in PINS}


def pin_card(p) -> str:
    bg, tag = TINTS[p["tint"]]
    title = f"<h4>{esc(p['title'])}</h4>" if p["title"] else ""
    cls = ("pin" if p["title"] else "pin quote") + (" dark" if p["tint"] == "lilac" else "")
    return (f'<div class="{cls}" style="background:{bg};min-height:{p["h"]}px"><div class="em">{p["emoji"]}</div>'
            f'{title}<p>{esc(p["body"])}</p><span class="tag" style="color:{tag}">{esc(p["cat"])}</span></div>')


def toggle_save(pid):
    if pid in ss.saved:
        ss.saved.remove(pid)
    else:
        ss.saved.append(pid)


def goto(key):
    ss.page = PAGE_BY_KEY[key]


def del_custom(pid):
    ss.custom_pins = [c for c in ss.custom_pins if c["id"] != pid]
    if pid in ss.saved:
        ss.saved.remove(pid)


def pin_actions(p, prefix, custom=False):
    c1, c2 = st.columns(2)
    saved = p["id"] in ss.saved
    c1.button("✓ Saved" if saved else "📌 Save", key=f"{prefix}s{p['id']}", on_click=toggle_save,
              args=(p["id"],), use_container_width=True, type="secondary" if saved else "primary")
    if custom:
        c2.button("Delete", key=f"{prefix}d{p['id']}", on_click=del_custom, args=(p["id"],), use_container_width=True)
    elif p.get("action"):
        c2.button("Try it", key=f"{prefix}t{p['id']}", on_click=goto, args=(p["action"],), use_container_width=True)
    if p.get("steps"):
        with st.expander("How to"):
            for n, s in enumerate(p["steps"], 1):
                st.markdown(f"{n}. {s}")


def render_masonry(pins, prefix, custom_ids=()):
    cols = st.columns(3, gap="medium")
    heights, buckets = [0, 0, 0], [[], [], []]
    for p in pins:                       # greedy: drop each pin into the shortest column
        i = heights.index(min(heights))
        buckets[i].append(p)
        heights[i] += p["h"] + 70
    for col, bucket in zip(cols, buckets):
        with col:
            for p in bucket:
                st.markdown(pin_card(p), unsafe_allow_html=True)
                pin_actions(p, prefix, p["id"] in custom_ids)


WIDE = "<style>.block-container {max-width:1200px !important;}</style>"


def page_discover():
    st.markdown(WIDE, unsafe_allow_html=True)
    who = f", {esc(ss.name)}" if ss.name else ""
    st.markdown(f'<div class="feedhead"><h1>Find what helps{who}</h1>'
                f'<p>Save ideas to your board. Tap Try it to jump into the matching tool.</p></div>',
                unsafe_allow_html=True)
    q = st.text_input("Search", placeholder="Search calm, exams, focus, sleep…", label_visibility="collapsed",
                      key="pinsearch")
    cats = ["All", "Calm", "Focus", "Study", "Self-care", "Motivation", "Saved"]
    if hasattr(st, "pills"):
        cat = st.pills("Browse", cats, default="All", label_visibility="collapsed") or "All"
    else:
        cat = st.radio("Browse", cats, horizontal=True, label_visibility="collapsed")
    allp = PINS + ss.custom_pins
    pins = [p for p in allp if p["id"] in ss.saved] if cat == "Saved" else \
           [p for p in PINS if cat == "All" or p["cat"] == cat]
    if q.strip():
        k = q.strip().lower()
        pins = [p for p in pins if k in (p["title"] + " " + p["body"] + " " + p["cat"]).lower()]
    elif cat == "All":                   # a fresh order every day
        off = dt.date.today().toordinal() % len(pins)
        pins = pins[off:] + pins[:off]
    if not pins:
        note("Nothing here yet. Try another word, or save a few pins first." if cat == "Saved"
             else "No pins match that search. Try a broader word like calm or study.")
        return
    render_masonry(pins, "d", {c["id"] for c in ss.custom_pins})


def page_board():
    st.markdown(WIDE, unsafe_allow_html=True)
    st.title("My board")
    st.caption("Everything that keeps you going, in one place: pins you saved and notes you wrote.")
    with st.expander("➕ Add your own pin"):
        with st.form("newpin", clear_on_submit=True):
            c1, c2 = st.columns([1, 2])
            emo = c1.selectbox("Icon", ["💛", "🌿", "🎯", "📚", "🏆", "🌈", "☕", "🤝", "✨"])
            tint = c2.selectbox("Colour", list(TINTS))
            title = st.text_input("Title (optional)", max_chars=60)
            body = st.text_area("Your note, goal or favourite line", max_chars=240)
            if st.form_submit_button("Pin it", type="primary") and (title.strip() or body.strip()):
                if not (guard(title, country) or guard(body, country)):
                    pid = f"u{int(time.time() * 1000)}"
                    ss.custom_pins.append(dict(id=pid, cat="My note", emoji=emo, title=title.strip(),
                                               body=body.strip(), tint=tint, h=min(300, 140 + len(body) // 2),
                                               action=None, steps=None))
                    ss.saved.append(pid)
                    st.rerun()
    lookup = {**PIN_BY_ID, **{c["id"]: c for c in ss.custom_pins}}
    pins = [lookup[i] for i in reversed(ss.saved) if i in lookup]
    if not pins:
        note("Your board is empty. Save pins from **Discover**, or add a note above.")
        return
    st.markdown(f"**{len(pins)}** pins")
    render_masonry(pins, "b", {c["id"] for c in ss.custom_pins})


{"Discover": page_discover, "My board": page_board, "Today": page_today, "Talk to Sage": page_talk,
 "Focus": page_focus, "Reframe": page_reframe, "Breathe": page_breathe, "Insights": page_insights,
 "Study tools": page_study, "Journal": page_journal, "Safety & help": page_safety}[page.split(" ", 1)[1]]()

save_data()