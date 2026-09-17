"""
theme.py — Design system for the Social Listening dashboard.

This module owns ONLY presentation: color tokens, typography, CSS, and small
HTML/Plotly helper functions. It does not read the CSV/parquet data and does
not contain any business or ML logic — app.py is responsible for that and
simply calls into these helpers to render it.
"""

import textwrap
import streamlit as st

# ============================================================================
# DESIGN TOKENS
# ============================================================================

COLORS = {
    # Surfaces
    "bg": "#F6F5F1",            # warm off-white app background
    "surface": "#FFFFFF",       # card / panel background
    "surface_alt": "#F0EEE7",   # subtle alternate surface (chips, table stripes)
    "border": "#E4E1D8",        # hairline borders
    "grid": "#EDEBE3",          # chart gridlines

    # Brand
    "navy": "#121B2C",          # deep navy — sidebar, primary text on light bg
    "navy_light": "#1E2C45",    # sidebar hover / secondary panels on navy
    "navy_soft": "#2A3B58",     # sidebar borders / dividers on navy
    "accent": "#3559C9",        # professional blue — active states, links, primary data
    "accent_soft": "#E9EDFB",   # accent tint for chips / soft backgrounds

    # Text
    "text": "#161E2B",
    "text_muted": "#57616E",        # secondary text on light surfaces (AA contrast)
    "text_faint": "#828C99",        # tertiary text, still readable
    "sidebar_text": "#D7DCE6",
    "sidebar_text_dim": "#96A1B5",

    # Data semantics — used consistently everywhere sentiment is shown
    "positive": "#1F8A5E",
    "positive_soft": "#E4F4EC",
    "negative": "#C0392B",
    "negative_soft": "#FBEAE7",
    "neutral_data": "#5C6B7A",

    # Sidebar action buttons — deliberately distinct from the semantic
    # sentiment colors above so "reset" and "export" read as actions, not
    # as data. Hover shades are ~12% darker for a clear pressed state.
    "success": "#10B981",
    "success_hover": "#0EA271",
    "danger": "#EF4444",
    "danger_hover": "#DC2626",
}

# Professional categorical palette for NON-sentiment multi-category charts
# (models, topics). Green/red are reserved for positive/negative only.
CATEGORICAL = ["#3559C9", "#5C6B7A", "#8C5FA8", "#B8873A", "#2E7D8C", "#8C3B4A"]

FONT_STACK = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

CHART_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}


def rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#RRGGBB' color to an 'rgba(r,g,b,a)' string.

    Plotly color properties (e.g. fillcolor) do not accept 8-digit hex
    (hex + alpha) — only 6-digit hex, rgb()/rgba(), hsl()/hsla(), or named
    CSS colors. Use this instead of string-concatenating an alpha suffix
    onto a hex color when building Plotly figures.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def raw_html(content: str):
    """st.markdown(..., unsafe_allow_html=True) for multi-line HTML blocks.

    Streamlit's markdown parser treats a line indented 4+ spaces as an
    indented code block, so a triple-quoted HTML string written inside a
    function body (and therefore indented) renders as literal escaped tags
    instead of real HTML. Dedenting first keeps the block starting at
    column 0 so it's parsed as raw HTML.
    """
    st.markdown(textwrap.dedent(content).strip("\n"), unsafe_allow_html=True)


# ============================================================================
# GLOBAL CSS
# ============================================================================

def inject_css():
    raw_html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,400,0,0');

    html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
    .stApp {{ background-color: {COLORS['bg']}; }}
    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
    .block-container {{ padding-top: 0.75rem; padding-bottom: 1.1rem; max-width: 100%;
                         padding-left: 1.6rem; padding-right: 1.6rem; }}

    /* Compact the vertical rhythm of the MAIN content area (not the sidebar,
       and not inside panel/KPI cards — those have their own tighter gap
       rules below with equal-or-higher CSS specificity, and this rule is
       declared earlier in the stylesheet so source order also favors them
       when specificity ties). Shrinks the default ~1rem Streamlit puts
       between stacked blocks (KPI row, chart rows, empty st.write("") spacers). */
    section[data-testid="stMain"] div[data-testid="stVerticalBlock"] {{ gap: 0.75rem; }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {{ gap: 1rem; }}
    section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] p:empty {{
        margin: 0; line-height: 0.4rem;
    }}
    /* KPI row (wrapped in st.container(key="kpi_row") on every page that has
       one) needs a clearly larger gap before the chart row below it than the
       generic inter-block gap above — this is the fix for KPI cards reading
       as glued to the charts underneath them. */
    div[class*="st-key-kpi_row"] {{ margin-bottom: 16px; }}

    .material-symbols-outlined {{
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 20;
        vertical-align: middle; line-height: 1;
    }}

    h1, h2, h3, h4 {{ color: {COLORS['text']}; font-family: {FONT_STACK}; }}
    p, span, div {{ font-family: {FONT_STACK}; }}

    /* ---------------------------------------------------------------- */
    /* Sidebar — compact, no-scroll layout                              */
    /* ---------------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['navy']};
        border-right: 1px solid {COLORS['navy_soft']};
    }}
    /* Sidebar header used by Streamlit for the collapse control.
       The sidebar is permanently expanded, so remove the header itself;
       hiding only the button leaves a large empty block at the top. */
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
        display: none !important;
    }}

    section[data-testid="stSidebar"] > div {{
        padding-top: 0 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
        padding-left: 0.65rem !important; padding-right: 0.65rem !important;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {{ color: {COLORS['sidebar_text']}; font-size: 0.88rem; }}

    /* Vertical rhythm between stacked sidebar widgets. 5px (the original
       spec) turned out too tight — Streamlit's own widgets (esp. the range
       slider, which needs headroom above the track for its floating
       current-value bubble) rely on some of that space internally, so
       squeezing every container to 0 margin made neighboring widgets visibly
       collide/overlap. 10px keeps the sidebar compact while giving each
       widget room to render itself correctly. */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{ gap: 10px; }}
    section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {{ margin-bottom: 0; }}

    /* Safety net: if content still slightly overflows on a shorter viewport,
       keep it scrollable via wheel/trackpad but hide the visible scrollbar
       chrome, rather than showing a bar. */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        scrollbar-width: none; /* Firefox */
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {{
        display: none; width: 0; height: 0; /* Chrome / Safari / Edge */
    }}

    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSelectbox label {{
        color: {COLORS['sidebar_text_dim']} !important;
        font-weight: 600; font-size: 0.76rem; letter-spacing: .2px; margin-bottom: 4px;
    }}
    /* The range slider draws a floating "current value" bubble above its
       track — give the widget itself extra top headroom so that bubble
       doesn't render on top of the "Khoảng ngày" label above it. */
    section[data-testid="stSidebar"] .stSlider {{ margin-top: 4px; padding-top: 14px; }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] {{ margin-top: 2px; }}
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
        background-color: {COLORS['accent']} !important; border-radius: 5px; font-size: 0.78rem;
        min-height: 24px !important; margin: 2px 4px 2px 0 !important;
    }}
    section[data-testid="stSidebar"] .stMultiSelect > div,
    section[data-testid="stSidebar"] .stSelectbox > div {{ margin-top: 2px; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {COLORS['navy_light']}; border-color: {COLORS['navy_soft']};
        border-radius: 7px; color: {COLORS['sidebar_text']}; font-size: 0.84rem;
        min-height: 36px !important;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: {COLORS['navy_soft']}; margin: 6px 0; }}

    /* Date-range slider — accent blue instead of Streamlit's default red,
       covering the selected-range highlight, the thumb fill/border and its
       focus ring (all matched with !important since BaseWeb applies its own
       inline theme colors that would otherwise win). */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div {{
        background: {COLORS['navy_soft']} !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div > div {{
        background-color: {COLORS['accent']} !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] [data-testid="stSliderTrackHighlight"] {{
        background-color: {COLORS['accent']} !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {COLORS['accent']} !important;
        border-color: {COLORS['accent']} !important;
        box-shadow: 0 0 0 3px {rgba(COLORS['accent'], 0.28)} !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="slider"] [data-testid="stTickBar"] div {{
        color: {COLORS['sidebar_text_dim']} !important;
    }}
    /* The floating "current value" bubble that appears above each thumb
       while dragging/focused — best-effort override since Streamlit doesn't
       expose a stable data-testid for it; targets the small text node
       BaseWeb renders as a sibling of the thumb. */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] div {{
        background-color: {COLORS['accent']} !important; color: #FFFFFF !important;
        border-color: {COLORS['accent']} !important;
    }}

    /* Sidebar brand row — plain "Social Listening" wordmark, no button (the
       sidebar no longer collapses, so there's nothing left to toggle). The
       icon + text themselves are wrapped in their own flex div in app.py;
       this just adds the divider/spacing around that row. */
    div[class*="st-key-sb_brand_row"] {{
        padding: 2px 0 8px 0; border-bottom: 1px solid {COLORS['navy_soft']}; margin-bottom: 10px; margin-top: 15px;
    }}
    .sb-brand-title {{ font-size: 1rem; font-weight: 700; color: #FFFFFF; line-height: 1.25; }}

    .sb-section-label {{
        font-size: 0.68rem; font-weight: 700; color: {COLORS['sidebar_text_dim']};
        letter-spacing: .4px; margin: 2px 0 4px 2px;
    }}

    /* Nav "cards" — the 4 main pages are the sidebar's primary visual focus.
       Sized so every label (incl. the longest, "Thử dự đoán trực tiếp")
       stays on one line without wrapping. */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] div.stButton {{ margin-bottom: 0; }}
    section[data-testid="stSidebar"] button {{
        font-size: 0.82rem !important; font-weight: 500 !important;
        border-radius: 8px !important; padding: 6px 10px !important;
        justify-content: flex-start !important; gap: 8px;
        white-space: nowrap !important; overflow: hidden !important;
        transition: background-color .12s ease;
    }}
    div[class*="st-key-nav_"] button {{
        font-size: 0.85rem !important; font-weight: 600 !important;
        padding: 9px 11px !important; gap: 10px !important;
        white-space: nowrap !important;
    }}
    div[class*="st-key-nav_"] button p {{ white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }}
    div[class*="st-key-nav_"] button span[data-testid="stIconMaterial"] {{ font-size: 18px !important; flex-shrink: 0; }}
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
        background-color: transparent !important; color: {COLORS['sidebar_text']} !important;
        border: 1px solid transparent !important;
    }}
    section[data-testid="stSidebar"] button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
        background-color: {COLORS['navy_light']} !important; color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
        background-color: {COLORS['navy_light']} !important; color: #FFFFFF !important;
        border: 1px solid {COLORS['navy_soft']} !important;
        border-left: 3px solid {COLORS['accent']} !important;
        box-shadow: none !important; font-weight: 700 !important;
    }}
    div[class*="st-key-nav_"] button[kind="primary"] {{ padding-left: 8px !important; }}

    /* "Đặt lại bộ lọc" — success green, stands out from the neutral nav
       buttons above it. */
    section[data-testid="stSidebar"] div[class*="st-key-reset_filters"] button {{
        background-color: {COLORS['success']} !important; border: 1px solid {COLORS['success']} !important;
        color: #FFFFFF !important; font-weight: 700 !important; justify-content: center !important;
    }}
    section[data-testid="stSidebar"] div[class*="st-key-reset_filters"] button:hover {{
        background-color: {COLORS['success_hover']} !important; border-color: {COLORS['success_hover']} !important;
    }}

    /* "Tải dữ liệu đã lọc (CSV)" — danger red, visually distinct as an
       export/leave-the-app action. */
    section[data-testid="stSidebar"] div[class*="st-key-export_csv"] button {{
        background-color: {COLORS['danger']} !important; border: 1px solid {COLORS['danger']} !important;
        color: #FFFFFF !important; font-weight: 700 !important; justify-content: center !important;
    }}
    section[data-testid="stSidebar"] div[class*="st-key-export_csv"] button:hover {{
        background-color: {COLORS['danger_hover']} !important; border-color: {COLORS['danger_hover']} !important;
    }}
    /* Small, deliberate gap so the export button doesn't read as glued to
       the dataset-info card right above it. */
    div[class*="st-key-export_csv"] {{ margin-top: 10px; }}

    /* Dataset info card — small bordered "glass" card instead of a bare
       icon + text row. */
    .sb-dataset-card {{
        display:flex; gap:8px; align-items:center; padding: 8px 9px;
        border-radius: 10px; border: 1px solid {COLORS['navy_soft']};
        background: linear-gradient(180deg, {rgba('#FFFFFF', 0.05)}, {rgba('#FFFFFF', 0.02)});
    }}
    .sb-dataset-icon-wrap {{
        width: 26px; height: 26px; border-radius: 7px; flex-shrink: 0;
        background: {rgba(COLORS['accent'], 0.22)};
        display:flex; align-items:center; justify-content:center;
    }}
    .sb-dataset-text {{ font-size: 0.76rem; color: {COLORS['sidebar_text_dim']}; line-height: 1.45; }}
    .sb-dataset-text b {{ color: #FFFFFF; font-weight: 600; }}

    /* ---------------------------------------------------------------- */
    /* Page header                                                      */
    /* ---------------------------------------------------------------- */
    .page-header {{
        display:flex; justify-content:space-between; align-items:flex-end;
        border-bottom: 1px solid {COLORS['border']}; padding-bottom: 8px; margin-bottom: 10px;
    }}
    .page-title {{ font-size: 1.55rem; font-weight: 800; color: {COLORS['text']}; letter-spacing: -.3px; }}
    .page-subtitle {{ font-size: 0.92rem; color: {COLORS['text_muted']}; margin-top: 2px; }}
    .page-header-right {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}

    .chip {{
        display:inline-flex; align-items:center; gap:6px; background: {COLORS['surface']};
        border: 1px solid {COLORS['border']}; border-radius: 999px; padding: 6px 13px;
        font-size: 0.88rem; color: {COLORS['text_muted']}; font-weight: 500; white-space: nowrap;
    }}
    .chip .material-symbols-outlined {{ font-size: 17px; color: {COLORS['text_faint']}; }}

    /* ---------------------------------------------------------------- */
    /* KPI cards                                                        */
    /* ---------------------------------------------------------------- */
    .kpi-card {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 10px;
        padding: 9px 12px; display:flex; align-items:center; gap: 10px;
        height: 58px; box-sizing: border-box;
    }}
    .kpi-icon {{
        width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
        display:flex; align-items:center; justify-content:center;
    }}
    .kpi-value {{ font-size: 1.32rem; font-weight: 800; color: {COLORS['text']}; line-height:1.05; }}
    .kpi-label {{ font-size: 0.8rem; color: {COLORS['text_muted']}; margin-top: 1px; font-weight: 500; }}
    .kpi-context {{ font-size: 0.74rem; color: {COLORS['text_faint']}; margin-top: 1px; }}

    /* ---------------------------------------------------------------- */
    /* Panels (chart / content cards)                                   */
    /* Real st.container(key=...) wrappers styled via the stable        */
    /* "st-key-<key>" class Streamlit adds to the container's outer div */
    /* — NOT raw HTML div tags spanning multiple st.markdown() calls,   */
    /* which Streamlit's HTML sanitizer treats as separate fragments    */
    /* and escapes/strips (visible stray "</div>" text + widgets        */
    /* rendering outside the intended card).                            */
    /* ---------------------------------------------------------------- */
    div[class*="st-key-panel_"] {{
        background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 10px;
        padding: 10px 13px 12px 13px; margin-bottom: 16px;
    }}
    /* Kill the default gap Streamlit puts between children inside a container —
       our own margins on .panel-head / charts / notes control spacing instead. */
    div[class*="st-key-panel_"] div[data-testid="stVerticalBlock"] {{ gap: 0.25rem; }}

    .panel-head {{
        display:flex; justify-content:space-between; align-items:flex-start;
        padding-bottom: 7px; border-bottom: 1px solid {COLORS['grid']}; margin-bottom: 3px;
    }}
    .panel-title {{ font-weight: 700; font-size: 0.98rem; color: {COLORS['text']}; }}
    .panel-subtitle {{ font-size: 0.82rem; color: {COLORS['text_muted']}; margin-top: 2px; }}
    .panel-badge {{
        font-size: 0.76rem; font-weight: 600; color: {COLORS['accent']}; background: {COLORS['accent_soft']};
        padding: 3px 9px; border-radius: 999px; white-space: nowrap;
    }}
    /* panel_note: padding is applied but exactly offset by a matching negative
       horizontal margin, so the text still lines up flush-left with the panel
       title/chart above it (both sit on the panel's own 13px inset) instead of
       reading as extra-indented. The padding still does real work: it creates
       genuine breathing room above the text (separating it from the chart) and
       below it (a buffer before the card's own bottom padding — so it never
       touches the card edge), matching the standalone padding: 8px 12px spec
       while keeping left-alignment correct. */
    .panel-note {{
        font-size: 0.82rem; color: #57616E; line-height: 1.5;
        padding: 8px 12px; margin: 2px -12px -2px -12px;
    }}

    /* Empty / info state */
    .empty-state {{
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap: 6px; padding: 18px 20px; color: {COLORS['text_muted']}; text-align:center;
    }}
    .empty-state .material-symbols-outlined {{ font-size: 32px; color: {COLORS['text_faint']}; }}
    .empty-state-text {{ font-size: 0.95rem; }}

    /* Prediction result */
    .predict-result {{ text-align:center; padding: 6px 0 4px 0; }}
    .predict-badge {{
        display:inline-flex; align-items:center; gap:7px; padding: 8px 18px; border-radius: 999px;
        font-size: 1.15rem; font-weight: 700; margin-top: 4px;
    }}
    .predict-confidence {{ font-size: 0.95rem; color: {COLORS['text_muted']}; margin-top: 9px; }}

    /* Streamlit tweaks: buttons in main area, dataframe, text area */
    div[data-testid="stTextArea"] textarea {{
        border-radius: 8px; border-color: {COLORS['border']}; font-family: {FONT_STACK};
    }}
    .stDataFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; overflow:hidden; }}

    /* Primary button (main content area) */
    div[data-testid="stAppViewContainer"] button[kind="primary"],
    div[data-testid="stAppViewContainer"] button[data-testid="stBaseButton-primary"] {{
        background-color: {COLORS['navy']} !important; border: 1px solid {COLORS['navy']} !important;
        border-radius: 8px !important; font-weight: 600 !important;
    }}
    div[data-testid="stAppViewContainer"] button[kind="primary"]:hover {{
        background-color: {COLORS['accent']} !important; border-color: {COLORS['accent']} !important;
    }}
    </style>
    """)


# NOTE: sidebar_collapse_css(collapsed) used to live here — it shrank the
# sidebar to an icon rail when collapsed. The sidebar no longer collapses
# (it's always expanded per spec), so that function, its CSS, and the
# `collapsed` / `sidebar_collapsed` session state that drove it from app.py
# were all removed together.


# ============================================================================
# ICON HELPER (Material Symbols Outlined — one consistent icon system)
# ============================================================================

def icon(name: str, color: str = "inherit", size: int = 18) -> str:
    return f'<span class="material-symbols-outlined" style="font-size:{size}px; color:{color};">{name}</span>'


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def page_header(title: str, subtitle: str | None, chips: list[str] | None = None):
    right = "".join(chips) if chips else ""
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    raw_html(f"""
    <div class="page-header">
        <div>
            <div class="page-title">{title}</div>{sub_html}
        </div>
        <div class="page-header-right">{right}</div>
    </div>
    """)


def chip(icon_name: str, text: str) -> str:
    return f'<div class="chip">{icon(icon_name, size=15)}<span>{text}</span></div>'


def kpi_card(icon_name: str, value: str, label: str, tone: str = "neutral", context: str | None = None):
    tone_map = {
        "neutral": COLORS["navy"],
        "positive": COLORS["positive"],
        "negative": COLORS["negative"],
        "accent": COLORS["accent"],
    }
    c = tone_map.get(tone, COLORS["navy"])
    # Built as a conditional list joined with no embedded blank lines: an empty
    # substitution left on its own line (e.g. when context=None) leaves a blank
    # line immediately followed by a 4-space-indented closing tag, which Markdown
    # parses as an indented code block — showing literal "</div>" text instead of
    # closing the tag (see raw_html() docstring for the general version of this).
    ctx_html = f'<div class="kpi-context">{context}</div>' if context else ""
    raw_html(f"""
    <div class="kpi-card">
        <div class="kpi-icon" style="background:{c}17; color:{c};">{icon(icon_name, c, 18)}</div>
        <div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>{ctx_html}
        </div>
    </div>
    """)


_panel_seq = {"n": 0}
_panel_stack: list = []


def panel_start(title: str, subtitle: str | None = None, badge: str | None = None):
    """Open a styled 'card' panel that real Streamlit widgets/charts can be placed inside.

    Uses a genuine st.container(key=...) rather than opening an HTML <div> in one
    st.markdown() call and closing it in a later one: Streamlit sanitizes each
    st.markdown(unsafe_allow_html=True) call as an independent HTML fragment, so a
    tag left open across calls gets escaped/stripped (visible stray "</div>" text)
    and any widgets rendered "in between" end up as siblings in the DOM, not
    children of the div — i.e. outside the visual card. st.container(key=...) adds
    a stable "st-key-<key>" class to its wrapper div (see CSS above) so the whole
    container can be styled as one real, correctly-nested card.
    """
    _panel_seq["n"] += 1
    key = f"panel_{_panel_seq['n']}"
    container = st.container(key=key)
    container.__enter__()
    _panel_stack.append(container)

    # Single-line HTML (no embedded newlines) on purpose: when subtitle/badge is
    # None, the substituted "" previously sat alone on its own template line,
    # leaving a blank line immediately followed by an indented closing </div> —
    # which Markdown parses as an indented code block (literal "</div>" text
    # instead of a closed tag). Keeping it all on one line sidesteps that
    # class of bug entirely instead of relying on exact indentation levels.
    sub_html = f'<div class="panel-subtitle">{subtitle}</div>' if subtitle else ""
    badge_html = f'<div class="panel-badge">{badge}</div>' if badge else ""
    raw_html(
        f'<div class="panel-head"><div><div class="panel-title">{title}</div>'
        f'{sub_html}</div>{badge_html}</div>'
    )


def panel_note(text: str):
    st.markdown(f'<div class="panel-note">{text}</div>', unsafe_allow_html=True)


def panel_end():
    container = _panel_stack.pop()
    container.__exit__(None, None, None)


def empty_state(text: str, icon_name: str = "search_off"):
    raw_html(f"""
    <div class="empty-state">
        {icon(icon_name, COLORS['text_faint'], 30)}
        <div class="empty-state-text">{text}</div>
    </div>
    """)


# ============================================================================
# CHART STYLING
# ============================================================================

def style_chart(fig, height: int = 200, **layout_kwargs):
    """Apply consistent typography, gridlines and chrome to a Plotly figure."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=COLORS["text"], size=12.5),
        margin=dict(l=12, r=12, t=10, b=16),
        height=height,
        # Explicit legend font color — relying only on the figure-wide `font`
        # above worked most of the time, but call sites that pass their own
        # `legend=dict(...)` (for orientation/position) were replacing this
        # entire dict rather than merging with it, so a caller-supplied
        # legend could silently lose the readable dark text color. Deep-merge
        # below fixes that instead of trusting dict.update().
        legend=dict(font=dict(color=COLORS["text"], size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=COLORS["navy"], font=dict(color="#FFFFFF", size=12.5, family=FONT_STACK),
                         bordercolor=COLORS["navy"]),
    )
    if "legend" in layout_kwargs:
        base["legend"] = {**base["legend"], **layout_kwargs.pop("legend")}
    if "margin" in layout_kwargs:
        base["margin"] = {**base["margin"], **layout_kwargs.pop("margin")}
    base.update(layout_kwargs)
    fig.update_layout(**base)

    # Axis title font applied via update_xaxes/update_yaxes AFTER
    # update_layout(**base) above — not before. update_layout(xaxis_title=...)
    # (used by callers via the xaxis_title/yaxis_title kwargs baked into
    # layout_kwargs) replaces the *entire* axis "title" object rather than
    # merging into it, so setting title_font earlier and xaxis_title later
    # silently wiped the font back to Plotly's own faint default — which is
    # exactly why axis titles were reading as barely visible. Doing it in
    # this order (title text first, font last) is what actually sticks.
    axis_title_font = dict(family=FONT_STACK, size=12, color=COLORS["text_muted"])
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor=COLORS["border"],
                      tickfont=dict(color=COLORS["text_muted"], size=11.5),
                      title_font=axis_title_font)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor=COLORS["border"],
                      tickfont=dict(color=COLORS["text_muted"], size=11.5),
                      title_font=axis_title_font)
    return fig