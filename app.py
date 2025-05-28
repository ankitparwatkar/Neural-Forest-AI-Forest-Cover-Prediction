# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import random

# Page configuration
st.set_page_config(
    page_title="🌲 NeuralForest AI",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Forest styling with dynamic title effects
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Ubuntu+Mono&display=swap');
    
    .main {{
        background: #0a1f0a;
        color: #e8f5e9;
        position: relative;
        overflow: hidden;
        padding-bottom: 100px !important;
    }}
    
    [data-testid="stSidebar"] {{
        background: #1b3b1b !important;
        border-right: 2px solid #4CAF50;
        box-shadow: 5px 0 15px rgba(76, 175, 80, 0.2);
    }}
    
    .sidebar .sidebar-content {{
        background: transparent !important;
        color: #e8f5e9 !important;
        font-family: 'Quicksand', sans-serif !important;
    }}
    
    .nature-glow {{
        text-shadow: 0 0 10px #4CAF50;
        color: #e8f5e9;
    }}
    
    .leaf-border {{
        border: 2px solid #4CAF50;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
        border-radius: 8px;
    }}
    
    .forest-card {{
        background: rgba(11, 32, 11, 0.9);
        backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }}
    
    .forest-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(11, 32, 11, 0.9);
        padding: 1rem;
        text-align: center;
        border-top: 2px solid #4CAF50;
        box-shadow: 0 -5px 20px rgba(76, 175, 80, 0.2);
        font-family: 'Quicksand', sans-serif;
        z-index: 9999;
    }}
    
    .tooltip {{
        cursor: help;
        border-bottom: 1px dotted #4CAF50;
    }}
    
    /* DYNAMIC TITLE STYLES */
    .dynamic-title {{
        font-family: 'Quicksand', sans-serif;
        text-align: center;
        padding: 20px;
        animation: titleGlow 2s infinite alternate;
    }}
    
    @keyframes titleGlow {{
        0% {{
            text-shadow: 0 0 5px #4CAF50;
            transform: scale(1);
        }}
        100% {{
            text-shadow: 0 0 20px #4CAF50, 0 0 30px #4CAF50;
            transform: scale(1.02);
        }}
    }}
    
    /* BOLD PREDICTION STYLES */
    .predicted-type {{
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        color: #4CAF50 !important;
        text-shadow: 0 0 10px rgba(76, 175, 80, 0.7) !important;
        text-align: center;
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 8px;
        background: rgba(11, 32, 11, 0.7);
        display: inline-block;
    }}
    
    .prediction-highlight {{
        border-left: 4px solid #4CAF50;
        padding-left: 1rem;
        margin: 1.5rem 0;
    }}
    
    /* ANIMATED FOREST BACKGROUND */
    .forest-bg {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: -1;
        opacity: 0.05;
        background: 
            radial-gradient(circle at 20% 30%, rgba(76, 175, 80, 0.2) 0px, transparent 40px),
            radial-gradient(circle at 80% 70%, rgba(76, 175, 80, 0.2) 0px, transparent 40px);
        animation: forestMove 60s infinite linear;
    }}
    
    @keyframes forestMove {{
        0% {{ background-position: 0% 0%; }}
        100% {{ background-position: 100% 100%; }}
    }}
    
    </style>
    
    <div class='forest-bg'></div>
    
    <div class='forest-footer'>
        <a href='https://www.linkedin.com/in/ankitparwatkar' style='color: #4CAF50 !important; margin: 0 1rem; text-decoration: none;'>🌲 LINKEDIN</a>
        <a href='https://github.com/ankitparwatkar' style='color: #4CAF50 !important; margin: 0 1rem; text-decoration: none;'>🍃 GITHUB</a>
        <div style='margin-top: 0.5rem; color: #4CAF50; font-size: 0.9rem;'>
            © 2025 NEURALFOREST AI • CREATED BY ANKIT PARWATKAR
        </div>
    </div>
    """, unsafe_allow_html=True)

# Forest type information
FOREST_INFO = {
    1: {
        "name": "Spruce/Fir",
        "description": "High elevation coniferous forests dominated by Engelmann spruce and subalpine fir.",
        "flora": ["Engelmann Spruce", "Subalpine Fir", "Bluejoint Reedgrass"],
        "fauna": ["Elk", "Marten", "Clark's Nutcracker"],
        "image": "spruce.jpg"
    },
    2: {
        "name": "Lodgepole Pine",
        "description": "Fire-adapted forests with dense stands of lodgepole pine trees.",
        "flora": ["Lodgepole Pine", "Huckleberry", "Beargrass"],
        "fauna": ["Red Squirrel", "Mountain Bluebird", "Black Bear"],
        "image": "lodgepole.jpg"
    },
    3: {
        "name": "Ponderosa Pine",
        "description": "Open-canopied forests with large, fire-resistant ponderosa pines.",
        "flora": ["Ponderosa Pine", "Bitterbrush", "Sagebrush"],
        "fauna": ["Mule Deer", "Abert's Squirrel", "Mountain Lion"],
        "image": "ponderosa.jpg"
    },
    4: {
        "name": "Cottonwood/Willow",
        "description": "Riparian forests along streams with cottonwood and willow species.",
        "flora": ["Plains Cottonwood", "Willow", "Sedge"],
        "fauna": ["Beaver", "Yellow Warbler", "Moose"],
        "image": "cottonwood.jpg"
    },
    5: {
        "name": "Aspen",
        "description": "Deciduous forests dominated by quaking aspen trees.",
        "flora": ["Quaking Aspen", "Saskatoon Berry", "Wild Rose"],
        "fauna": ["Elk", "Ruffed Grouse", "Black Bear"],
        "image": "aspen.jpg"
    },
    6: {
        "name": "Douglas-Fir",
        "description": "Moist forests with tall Douglas-fir and western hemlock.",
        "flora": ["Douglas-Fir", "Western Hemlock", "Salal"],
        "fauna": ["Spotted Owl", "Flying Squirrel", "Black-tailed Deer"],
        "image": "douglas.jpg"
    },
    7: {
        "name": "Krummholz",
        "description": "Stunted subalpine forests shaped by harsh winds and cold.",
        "flora": ["Subalpine Fir", "Engelmann Spruce", "Alpine Willow"],
        "fauna": ["Pika", "White-tailed Ptarmigan", "Marmot"],
        "image": "krummholz.jpg"
    }
}

# Constants and model loading
CLASS_NAMES = {k: f"{v['name'].upper()} 🌲🌳🌿" for k, v in FOREST_INFO.items()}

NUMERICAL_COLS = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
    'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

CATEGORICAL_COLS = [f'Wilderness_Area{i}' for i in range(1, 5)] + \
                  [f'Soil_Type{i}' for i in range(1, 41)]

FEATURE_COLUMNS = NUMERICAL_COLS + CATEGORICAL_COLS

@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

model = load_model()
scaler = load_scaler()

def create_forest_chart(values, categories):
    fig = plt.figure(figsize=(8, 8), facecolor='#0a1f0a')
    ax = fig.add_subplot(111, polar=True)
    
    ax.set_facecolor('#0a1f0a')
    ax.spines['polar'].set_color('#4CAF50')
    ax.tick_params(axis='both', colors='#e8f5e9')
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    values += values[:1]
    ax.plot(angles, values, color='#4CAF50', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='#4CAF50', alpha=0.25)
    
    plt.xticks(angles[:-1], categories, color='#e8f5e9')
    plt.yticks(color='#e8f5e9')
    plt.ylim(0, 1)
    
    return fig

# Sidebar content
with st.sidebar:
    st.markdown("""
        <h1 class='nature-glow'>🌳 FOREST COVER AI</h1>
        <div style='border-top: 2px solid #4CAF50; margin: 1rem 0;'></div>
        <p style='font-size: 0.9rem;'>🖥 Advanced predictive model for forest cover type classification</p>
        
        <h3>📚 Guide</h3>
        <div class="tooltip">Hover over input labels for help</div>
        <ul>
            <li>Adjust sliders based on geographical data</li>
            <li>Use expanders for detailed inputs</li>
            <li>Click 'Analyze Ecosystem' for prediction</li>
        </ul>
        
        <h3>🌍 Why It Matters</h3>
        <p>Forest cover analysis helps in:</p>
        <ul>
            <li>Biodiversity conservation</li>
            <li>Wildfire prevention</li>
            <li>Climate change mitigation</li>
            <li>Ecosystem management</li>
        </ul>
    """, unsafe_allow_html=True)

def main():
    # Dynamic animated title
    st.markdown("<h1 class='dynamic-title'>🌳 NEURALFOREST ECOSYSTEM AI</h1>", unsafe_allow_html=True)
    
    # Introduction Section
    with st.expander("🌟 Welcome to NeuralForest Ecosystem AI", expanded=True):
        st.markdown("""
        **Explore forest ecosystems through AI-powered analysis.**  
        This tool helps you:
        - Predict forest cover types using cartographic data
        - Understand ecosystem characteristics
        - Visualize environmental relationships
        
        *Begin by adjusting the parameters below or use default values for a quick demo.*
        """)
    
    # Interactive Input Section
    with st.form("forest_form"):
        with st.container():
            st.markdown('<div class="forest-card leaf-border">', unsafe_allow_html=True)
            
            cols = st.columns(3)
            with cols[0]:
                user_input = {}
                user_input['Elevation'] = st.slider(
                    "⛰️ Elevation (meters)",
                    1850, 3850, 2500,
                    help="Height above sea level - affects temperature and precipitation"
                )
                user_input['Aspect'] = st.slider(
                    "🌄 Aspect (degrees)",
                    0, 360, 180,
                    help="Compass direction the slope faces - impacts sunlight exposure"
                )
                user_input['Slope'] = st.slider(
                    "📐 Slope (degrees)",
                    0, 60, 30,
                    help="Steepness of terrain - affects water runoff and soil erosion"
                )
            
            with cols[1]:
                user_input['Horizontal_Distance_To_Hydrology'] = st.slider(
                    "💧 Water Distance (meters)",
                    0, 1500, 300,
                    help="Horizontal distance to nearest water source"
                )
                user_input['Vertical_Distance_To_Hydrology'] = st.slider(
                    "🔼 Vertical Water (meters)",
                    -200, 600, 0,
                    help="Elevation difference to nearest water source"
                )
                user_input['Horizontal_Distance_To_Roadways'] = st.slider(
                    "🛣️ Road Distance (meters)",
                    0, 7000, 1000,
                    help="Distance to nearest man-made road"
                )
            
            with cols[2]:
                user_input['Hillshade_9am'] = st.slider(
                    "🌅 Morning Shade Index",
                    0, 255, 200,
                    help="Light conditions at 9 AM (0=dark, 255=bright)"
                )
                user_input['Hillshade_Noon'] = st.slider(
                    "🌞 Noon Shade Index",
                    0, 255, 220,
                    help="Light conditions at noon"
                )
                user_input['Hillshade_3pm'] = st.slider(
                    "🌇 Evening Shade Index",
                    0, 255, 200,
                    help="Light conditions at 3 PM"
                )

            # Advanced Parameters
            with st.expander("🔬 Advanced Parameters", expanded=False):
                cols_adv = st.columns(2)
                with cols_adv[0]:
                    user_input['Horizontal_Distance_To_Fire_Points'] = st.slider(
                        "🔥 Fire Distance (meters)",
                        0, 7000, 1500,
                        help="Distance to nearest historical fire occurrence"
                    )
                with cols_adv[1]:
                    user_input['wilderness_area'] = st.radio(
                        "🌲 Wilderness Area",
                        ["1", "2", "3", "4"],
                        horizontal=True,
                        help="Protected wilderness area designation"
                    )
                    user_input['soil_type'] = st.selectbox(
                        "🌱 Soil Type",
                        options=list(range(1, 41)),
                        help="USDA soil type classification"
                    )

            submitted = st.form_submit_button("🌿 Analyze Ecosystem", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

    # Results Section
    if submitted:
        with st.spinner("🌳 Analyzing forest patterns..."):
            try:
                features = pd.DataFrame([[0]*len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
                
                for col in NUMERICAL_COLS:
                    features[col] = user_input[col]
                
                features[f'Wilderness_Area{user_input["wilderness_area"]}'] = 1
                features[f'Soil_Type{user_input["soil_type"]}'] = 1
                
                features[NUMERICAL_COLS] = scaler.transform(features[NUMERICAL_COLS])
                
                prediction = model.predict(features)[0]
                
                st.success("✅ Analysis Complete!")
                
                # Results Container
                with st.container():
                    # Forest Type Card with bold prediction
                    with st.expander("🌲 FOREST COVER PREDICTION", expanded=True):
                        # Bold prediction display
                        st.markdown(
                            f"<div class='predicted-type'>PREDICTED FOREST TYPE: {CLASS_NAMES[prediction]}</div>", 
                            unsafe_allow_html=True
                        )
                        
                        col_info, col_img = st.columns([2, 1])
                        with col_info:
                            st.markdown(f"""
                                **ECOSYSTEM CHARACTERISTICS**  
                                <div class='prediction-highlight'>{FOREST_INFO[prediction]['description']}</div>
                                
                                **COMMON FLORA:**
                                - {FOREST_INFO[prediction]['flora'][0]}
                                - {FOREST_INFO[prediction]['flora'][1]}
                                - {FOREST_INFO[prediction]['flora'][2]}
                                
                                **TYPICAL FAUNA:**
                                - {FOREST_INFO[prediction]['fauna'][0]}
                                - {FOREST_INFO[prediction]['fauna'][1]}
                                - {FOREST_INFO[prediction]['fauna'][2]}
                            """, unsafe_allow_html=True)
                        with col_img:
                            st.image(FOREST_INFO[prediction]['image'], caption="Typical forest appearance")

                    # Visualization Section
                    with st.expander("📊 ENVIRONMENTAL ANALYSIS", expanded=True):
                        st.markdown("### ECOSYSTEM PROFILE RADAR CHART")
                        fig = create_forest_chart(
                            values=[
                                user_input['Elevation']/3850,
                                user_input['Slope']/60,
                                user_input['Horizontal_Distance_To_Hydrology']/1500,
                                user_input['Horizontal_Distance_To_Roadways']/7000,
                                user_input['Horizontal_Distance_To_Fire_Points']/7000
                            ],
                            categories=['Elevation', 'Slope', 'Water', 'Roads', 'Fire']
                        )
                        st.pyplot(fig)
                        st.markdown("""
                            **CHART GUIDE:**
                            - Shows relative values of key environmental factors
                            - Larger area indicates stronger presence/impact
                            - Compare different locations by running multiple analyses
                        """)

                    # Conservation Tips
                    with st.expander("💡 CONSERVATION RECOMMENDATIONS", expanded=True):
                        st.markdown(f"""
                        **FOR {FOREST_INFO[prediction]['name'].upper()} ECOSYSTEMS:**
                        <div class='prediction-highlight'>
                        - Maintain natural fire regimes
                        - Monitor invasive species
                        - Protect watershed areas
                        - Implement sustainable tourism practices
                        </div>
                        
                        *Learn more about conservation efforts at [Global Forest Watch](https://www.globalforestwatch.org/)*
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()