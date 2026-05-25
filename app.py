"""
🍽️ AI Calorie Calculator — Powered by Claude Vision
Snap a photo of any food and get instant calorie + nutrition breakdown.
"""

try:
    import pip_system_certs.wrapt_requests  # fix SSL cert on hotspot/restricted networks
except ImportError:
    pass

import streamlit as st
import anthropic
import base64
import json
import re
import os
import io
from datetime import date, datetime
from PIL import Image
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
APP_VERSION = "1.0"
DEFAULT_GOAL = 2000

SYSTEM_PROMPT = """You are an expert nutritionist and food scientist specialising in calorie estimation from food images.
Your role is to accurately identify and quantify every food item visible in the image and return a precise nutritional breakdown.

RULES:
1. Identify ALL items: mains, sides, sauces, drinks, garnishes — nothing is invisible.
2. Estimate portion sizes using visual cues: plate diameter (~25 cm standard), hands, utensils, glass size.
3. For cooking method: fried items add ~15-25% extra fat calories; grilled/steamed are base values.
4. Use NIN India tables for Indian food; USDA FoodData Central for Western food.
5. Give REAL numbers — do not round everything to 50. Be precise (e.g., 237 kcal not 250).
6. Set confidence: "high" (clearly identifiable), "medium" (likely but could vary), "low" (guessing).

OUTPUT: Respond with ONLY a valid JSON object — no markdown, no explanation outside JSON:
{
  "meal_name": "Short descriptive name",
  "cuisine_type": "Indian / Chinese / Italian / Fast Food / etc.",
  "food_items": [
    {
      "name": "Exact food item name",
      "quantity": "Estimated quantity with unit",
      "calories": 0,
      "protein_g": 0.0,
      "carbs_g": 0.0,
      "fat_g": 0.0,
      "fiber_g": 0.0,
      "confidence": "high"
    }
  ],
  "total_calories": 0,
  "total_protein_g": 0.0,
  "total_carbs_g": 0.0,
  "total_fat_g": 0.0,
  "total_fiber_g": 0.0,
  "health_score": 7,
  "health_rating": "Balanced",
  "health_notes": "One sentence summary of nutritional quality",
  "suggestions": ["Tip 1 to make this meal healthier", "Tip 2"],
  "estimation_notes": "Any important caveats about accuracy"
}

health_score: integer 1-10 (10 = most nutritious)
health_rating: one of — Excellent / Balanced / Moderate / Indulgent / Heavy

If the image is NOT food or too blurry to identify, return:
{"error": true, "message": "Explanation of why analysis failed"}"""

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0f172a; }
[data-testid="stHeader"]           { background:transparent; }

/* Metric cards */
[data-testid="metric-container"] {
    background:#1e293b; border:1px solid #334155;
    border-radius:14px; padding:18px 22px;
    box-shadow:0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:.8rem !important;
    font-weight:700 !important; text-transform:uppercase; letter-spacing:.05em; }
[data-testid="stMetricValue"] { color:#f1f5f9 !important; font-size:1.7rem !important; font-weight:700 !important; }

/* Buttons */
[data-testid="stFormSubmitButton"]>button, .stButton>button {
    background:linear-gradient(135deg,#10b981,#059669) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:700 !important;
    font-size:1rem !important; padding:10px 28px !important;
    transition:all .2s !important;
}
[data-testid="stFormSubmitButton"]>button:hover, .stButton>button:hover {
    opacity:.85 !important; transform:translateY(-1px) !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] { color:#94a3b8 !important; font-weight:600 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#10b981 !important;
    border-bottom:2px solid #10b981 !important; }

/* Upload / camera */
[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background:#1e293b; border:2px dashed #334155;
    border-radius:16px; padding:12px;
}

/* Food item card */
.food-card {
    background:#1e293b; border:1px solid #334155;
    border-radius:14px; padding:16px 20px; margin:8px 0;
}
.food-name  { font-size:1.05rem; font-weight:700; color:#f1f5f9; }
.food-qty   { font-size:.85rem; color:#94a3b8; margin-bottom:4px; }
.cal-badge  { background:#064e3b; color:#34d399; padding:4px 12px;
              border-radius:20px; font-weight:700; font-size:.95rem; display:inline-block; }

/* Score bar */
.score-bar-bg { background:#1e293b; border-radius:8px; height:12px;
                border:1px solid #334155; overflow:hidden; }

/* Suggestion chip */
.chip { background:#1e293b; border:1px solid #334155; border-radius:20px;
        padding:5px 14px; font-size:.88rem; color:#94a3b8;
        display:inline-block; margin:3px; }
hr { border-color:#334155 !important; }
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=92)
    return buf.getvalue()

def get_media_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {"jpg":"image/jpeg","jpeg":"image/jpeg",
            "png":"image/png","webp":"image/webp",
            "gif":"image/gif"}.get(ext, "image/jpeg")

def extract_json(text: str):
    """Robustly pull JSON from Claude's response."""
    text = text.strip()
    # Strip markdown code fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
    return None

def calorie_color(cal: int) -> str:
    if cal < 300:  return "#22c55e"
    if cal < 600:  return "#f59e0b"
    if cal < 900:  return "#f97316"
    return "#ef4444"

def score_color(score: int) -> str:
    if score >= 8: return "#22c55e"
    if score >= 5: return "#f59e0b"
    return "#ef4444"

def analyze_food(image_bytes: bytes, media_type: str, client: anthropic.Anthropic):
    """Call Claude vision API and return parsed JSON."""
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64",
                            "media_type": media_type,
                            "data": img_b64}},
                {"type": "text",
                 "text": ("Analyze this food image. Identify every item visible, "
                          "estimate portions accurately, and return the JSON nutrition breakdown.")}
            ]
        }]
    )
    # Extract text block (skip thinking blocks)
    raw = ""
    for block in response.content:
        if block.type == "text":
            raw = block.text
            break
    return extract_json(raw), raw

# ── Macro donut chart ─────────────────────────────────────────────────────────
def macro_chart(protein, carbs, fat):
    labels = ["Protein", "Carbs", "Fat"]
    values = [protein * 4, carbs * 4, fat * 9]  # kcal from each macro
    colors = ["#6366f1", "#f59e0b", "#ef4444"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#f1f5f9"),
        hovertemplate="<b>%{label}</b><br>%{value:.0f} kcal<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{int(protein+carbs+fat):.0f}g</b><br>macros",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#f1f5f9"),
        xref="paper", yref="paper"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=280,
    )
    return fig

# ── Results renderer ──────────────────────────────────────────────────────────
def render_results(data: dict, thumb: Image.Image):
    total_cal = data.get("total_calories", 0)
    total_p   = data.get("total_protein_g", 0)
    total_c   = data.get("total_carbs_g", 0)
    total_f   = data.get("total_fat_g", 0)
    total_fi  = data.get("total_fiber_g", 0)
    score     = data.get("health_score", 5)
    rating    = data.get("health_rating", "—")
    cuisine   = data.get("cuisine_type", "—")
    meal_name = data.get("meal_name", "Food Analysis")

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:16px;
            padding:20px 24px;margin-bottom:16px;'>
  <h2 style='color:#10b981;margin:0;font-size:1.6rem;'>🍽️ {meal_name}</h2>
  <p style='color:#64748b;margin:4px 0 0 0;font-size:.9rem;'>
    Cuisine: <b style='color:#94a3b8'>{cuisine}</b></p>
</div>""", unsafe_allow_html=True)

    # ── macro cards ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔥 Calories",  f"{total_cal} kcal")
    c2.metric("🥩 Protein",   f"{total_p:.1f} g")
    c3.metric("🌾 Carbs",     f"{total_c:.1f} g")
    c4.metric("🥑 Fat",       f"{total_f:.1f} g")
    c5.metric("🌿 Fiber",     f"{total_fi:.1f} g")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    # ── food items list ───────────────────────────────────────────────────────
    with left:
        st.markdown("##### 📋  Food Items Detected")
        items = data.get("food_items", [])
        for item in items:
            conf_icon = {"high":"✅","medium":"⚠️","low":"🔍"}.get(
                item.get("confidence","medium"), "⚠️")
            conf_col  = {"high":"#22c55e","medium":"#f59e0b","low":"#ef4444"}.get(
                item.get("confidence","medium"), "#f59e0b")
            st.markdown(f"""
<div class='food-card'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <div class='food-name'>{item.get("name","—")}</div>
      <div class='food-qty'>📏 {item.get("quantity","—")}</div>
    </div>
    <div style='text-align:right;'>
      <span class='cal-badge'>{item.get("calories",0)} kcal</span><br>
      <small style='color:{conf_col};font-size:.75rem;'>{conf_icon} {item.get("confidence","—")}</small>
    </div>
  </div>
  <div style='margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;'>
    <small style='color:#6366f1;'>P: <b>{item.get("protein_g",0):.1f}g</b></small>
    <small style='color:#f59e0b;'>C: <b>{item.get("carbs_g",0):.1f}g</b></small>
    <small style='color:#ef4444;'>F: <b>{item.get("fat_g",0):.1f}g</b></small>
    <small style='color:#22c55e;'>Fiber: <b>{item.get("fiber_g",0):.1f}g</b></small>
  </div>
</div>""", unsafe_allow_html=True)

    # ── right panel ───────────────────────────────────────────────────────────
    with right:
        # Thumbnail
        st.image(thumb, use_container_width=True,
                 caption="Analyzed image")

        # Macro chart
        st.markdown("##### 🥗  Macro Breakdown")
        if total_p + total_c + total_f > 0:
            st.plotly_chart(macro_chart(total_p, total_c, total_f),
                            use_container_width=True)

        # Health score
        st.markdown("##### 💚  Health Score")
        sc = score; sc_col = score_color(sc)
        pct = sc * 10
        st.markdown(f"""
<div style='margin:8px 0;'>
  <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
    <span style='color:{sc_col};font-weight:700;font-size:1.2rem;'>{sc}/10</span>
    <span style='color:{sc_col};font-weight:700;'>{rating}</span>
  </div>
  <div class='score-bar-bg'>
    <div style='width:{pct}%;height:100%;background:{sc_col};border-radius:8px;transition:width .5s;'></div>
  </div>
  <p style='color:#94a3b8;font-size:.85rem;margin-top:8px;'>{data.get("health_notes","")}</p>
</div>""", unsafe_allow_html=True)

    # ── suggestions ───────────────────────────────────────────────────────────
    suggestions = data.get("suggestions", [])
    if suggestions:
        st.markdown("---")
        st.markdown("##### 💡  Healthier Choices")
        chips = "".join(f"<span class='chip'>💡 {s}</span>" for s in suggestions)
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

    # ── estimation notes ─────────────────────────────────────────────────────
    notes = data.get("estimation_notes", "")
    if notes:
        st.caption(f"ℹ️  {notes}")

# ── Daily tracker ─────────────────────────────────────────────────────────────
def render_daily_log():
    log = st.session_state.get("meal_log", [])
    if not log:
        st.info("No meals logged today. Snap a photo to start tracking! 📸")
        return

    goal = st.session_state.get("daily_goal", DEFAULT_GOAL)
    total = sum(m["calories"] for m in log)
    remaining = goal - total
    pct = min(total / goal * 100, 100)
    bar_col = "#22c55e" if pct < 80 else "#f59e0b" if pct < 100 else "#ef4444"

    st.markdown(f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;margin-bottom:16px;'>
  <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
    <span style='color:#94a3b8;font-weight:700;text-transform:uppercase;font-size:.85rem;'>Daily Progress</span>
    <span style='color:{bar_col};font-weight:700;'>{total} / {goal} kcal</span>
  </div>
  <div class='score-bar-bg'>
    <div style='width:{pct:.1f}%;height:100%;background:{bar_col};border-radius:8px;'></div>
  </div>
  <p style='color:#64748b;font-size:.85rem;margin-top:6px;'>
    {"✅ Within goal" if remaining > 0 else "⚠️ Over goal"} —
    {abs(remaining)} kcal {"remaining" if remaining > 0 else "over"}
  </p>
</div>""", unsafe_allow_html=True)

    for i, meal in enumerate(reversed(log)):
        col1, col2, col3 = st.columns([3, 1, 0.5])
        with col1:
            st.markdown(f"**{meal['name']}**  "
                        f"<small style='color:#64748b;'>{meal['time']}</small>",
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span style='color:{calorie_color(meal['calories'])};font-weight:700;'>"
                        f"{meal['calories']} kcal</span>", unsafe_allow_html=True)
        with col3:
            if st.button("🗑️", key=f"del_{i}", help="Remove"):
                idx = len(log) - 1 - i
                st.session_state.meal_log.pop(idx)
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="AI Calorie Calculator",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    # Session state init
    if "meal_log"      not in st.session_state: st.session_state.meal_log = []
    if "last_result"   not in st.session_state: st.session_state.last_result = None
    if "last_image"    not in st.session_state: st.session_state.last_image = None
    if "daily_goal"    not in st.session_state: st.session_state.daily_goal = DEFAULT_GOAL
    if "log_date"      not in st.session_state: st.session_state.log_date = date.today()

    # Reset log on new day
    if st.session_state.log_date != date.today():
        st.session_state.meal_log = []
        st.session_state.log_date = date.today()

    # ── Sidebar settings ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️  Settings")
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Get your key from console.anthropic.com"
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        st.session_state.daily_goal = st.number_input(
            "Daily Calorie Goal (kcal)",
            min_value=1000, max_value=4000,
            value=st.session_state.daily_goal, step=50
        )
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown(
            "1. 📸 Upload or snap a photo\n"
            "2. 🤖 Claude AI identifies food\n"
            "3. 🔢 Get instant calorie breakdown\n"
            "4. 📊 Track daily intake"
        )
        st.markdown("---")
        st.caption("Powered by Claude Opus 4.7 Vision")

    # ── Header ───────────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([0.7, 0.3])
    with hcol1:
        st.markdown(
            "<h1 style='margin:0;background:linear-gradient(135deg,#10b981,#34d399);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "font-size:2.2rem;'>🍽️ AI Calorie Calculator</h1>"
            "<p style='color:#64748b;margin:2px 0 0 0;'>Upload any food photo → instant nutrition breakdown</p>",
            unsafe_allow_html=True
        )
    with hcol2:
        logged_today = len(st.session_state.meal_log)
        cal_today    = sum(m["calories"] for m in st.session_state.meal_log)
        st.metric("Today's Intake", f"{cal_today} kcal",
                  delta=f"{logged_today} meals logged")

    st.markdown("---")

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📸  Analyze Food", "📋  Today's Log"])

    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # Check API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("⚠️  Enter your Anthropic API key in the sidebar to start.", icon="🔑")
            st.stop()

        # ── Image input ──────────────────────────────────────────────────────
        st.markdown("#### Choose how to add your food photo:")
        method = st.radio("", ["📁  Upload Image", "📷  Use Camera"],
                          horizontal=True, label_visibility="collapsed")

        img_bytes, media_type, thumb = None, "image/jpeg", None

        if method == "📁  Upload Image":
            uploaded = st.file_uploader(
                "Drag & drop or click to upload",
                type=["jpg","jpeg","png","webp"],
                label_visibility="collapsed"
            )
            if uploaded:
                img_bytes  = uploaded.read()
                media_type = get_media_type(uploaded.name)
                thumb      = Image.open(io.BytesIO(img_bytes))

        else:
            photo = st.camera_input("Take a photo of your food",
                                    label_visibility="collapsed")
            if photo:
                img_bytes  = photo.read()
                media_type = "image/jpeg"
                thumb      = Image.open(io.BytesIO(img_bytes))

        # ── Analyze button ───────────────────────────────────────────────────
        if img_bytes:
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn, col_log = st.columns([1, 3])
            with col_btn:
                analyze_clicked = st.button("🔍  Analyze Calories", use_container_width=True)

            if analyze_clicked:
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

                with st.spinner("🤖  Claude is analyzing your food… (5–15 seconds)"):
                    try:
                        # Resize if too large
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        if max(pil_img.size) > 1536:
                            pil_img.thumbnail((1536, 1536), Image.LANCZOS)
                            img_bytes = pil_to_bytes(pil_img)

                        result, raw = analyze_food(img_bytes, media_type, client)

                        if result is None:
                            st.error("❌  Could not parse nutritional data. Raw response:")
                            st.code(raw[:1000])
                        elif result.get("error"):
                            st.error(f"❌  {result.get('message','Could not identify food.')}")
                        else:
                            st.session_state.last_result = result
                            st.session_state.last_image  = thumb
                            # Auto-add to log
                            st.session_state.meal_log.append({
                                "name":     result.get("meal_name","Meal"),
                                "calories": result.get("total_calories", 0),
                                "protein":  result.get("total_protein_g", 0),
                                "carbs":    result.get("total_carbs_g", 0),
                                "fat":      result.get("total_fat_g", 0),
                                "time":     datetime.now().strftime("%I:%M %p"),
                                "data":     result,
                            })
                            st.success("✅  Analysis complete — added to today's log!")

                    except anthropic.AuthenticationError:
                        st.error("❌  Invalid API key. Check your Anthropic API key in the sidebar.")
                    except Exception as e:
                        st.error(f"❌  Error: {str(e)}")

        # ── Show last result ─────────────────────────────────────────────────
        if st.session_state.last_result:
            st.markdown("---")
            st.markdown("#### 📊  Nutrition Analysis")
            thumb_show = st.session_state.last_image or thumb
            render_results(st.session_state.last_result, thumb_show)

        elif not img_bytes:
            # Placeholder
            st.markdown("""
<div style='background:#1e293b;border:2px dashed #334155;border-radius:20px;
            padding:60px;text-align:center;margin-top:20px;'>
  <p style='font-size:3rem;margin:0;'>📸</p>
  <h3 style='color:#94a3b8;margin-top:12px;'>Upload a photo of any food</h3>
  <p style='color:#475569;'>Works with Indian food, fast food, home cooking, restaurants — any cuisine!</p>
  <br>
  <div style='display:flex;justify-content:center;gap:20px;flex-wrap:wrap;'>
    <span class='chip'>🍛 Dal Makhani</span>
    <span class='chip'>🍝 Pasta</span>
    <span class='chip'>🍱 Thali</span>
    <span class='chip'>🍔 Burger</span>
    <span class='chip'>🥗 Salad</span>
    <span class='chip'>🍜 Noodles</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 📋  Today's Meal Log")
        st.caption(f"Date: {date.today().strftime('%d %B %Y')}")
        render_daily_log()

        if st.session_state.meal_log:
            st.markdown("---")
            # Summary metrics
            logs = st.session_state.meal_log
            total_p = sum(m["protein"] for m in logs)
            total_c = sum(m["carbs"]   for m in logs)
            total_f = sum(m["fat"]     for m in logs)
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Protein",  f"{total_p:.1f} g")
            s2.metric("Total Carbs",    f"{total_c:.1f} g")
            s3.metric("Total Fat",      f"{total_f:.1f} g")

            if st.button("🗑️  Clear Today's Log"):
                st.session_state.meal_log = []
                st.session_state.last_result = None
                st.rerun()

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#334155;font-size:.78rem;'>"
        "🍽️ AI Calorie Calculator  ·  Powered by Claude Opus 4.7 Vision  ·  "
        "Estimates may vary ±15% — use as a guide, not medical advice</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
