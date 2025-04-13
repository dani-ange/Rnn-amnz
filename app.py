import streamlit as st
import yfinance as yf
from streamlit_modal import Modal
import streamlit.components.v1 as components

# Set the page layout
st.set_page_config(layout="wide")

import matplotlib.pyplot as plt
import numpy as np
import base64
import pandas as pd
import time
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


if "framework" not in st.session_state:
    st.session_state.framework = "gen"
# Initialize state
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "show_overlay" not in st.session_state:
    st.session_state.show_overlay = False
if "model" not in st.session_state:
    st.session_state.model = "best_bilstm_model.h5"


# Loading  model
@st.cache_resource
def load_lstm_model(path):
    return load_model(path)


@st.cache_resource
def load_data():
    data = yf.download("AMZN", period="4y", multi_level_index=False)
    data.reset_index(inplace=True)
    return data


#################################################################################################


def predict_future_prices(
    df: pd.DataFrame, n_future_days: int, model_path: str = st.session_state.model
) -> tuple[plt.Figure, pd.DataFrame]:
    # Ensure DataFrame is sorted and clean
    df = df.sort_values("Date").dropna(subset=["Close"])
    df = df.reset_index(drop=True)

    # Scale data
    scaler = MinMaxScaler()
    prices = df["Close"].values.reshape(-1, 1)
    scaled_prices = scaler.fit_transform(prices)

    # Load model and get timesteps
    model = load_lstm_model(model_path)
    n_steps = model.input_shape[1]

    # --- Calculate validation error (historical residuals) ---
    X_hist, y_hist = [], []
    for i in range(n_steps, len(scaled_prices)):
        X_hist.append(scaled_prices[i - n_steps : i])
        y_hist.append(scaled_prices[i])
    X_hist = np.array(X_hist)
    y_hist = np.array(y_hist)

    # Predict historical values
    hist_predictions = model.predict(X_hist, verbose=0)

    # Calculate residuals (error)
    hist_prices_rescaled = scaler.inverse_transform(y_hist.reshape(-1, 1)).flatten()
    hist_preds_rescaled = scaler.inverse_transform(
        hist_predictions.reshape(-1, 1)
    ).flatten()
    residuals = hist_prices_rescaled - hist_preds_rescaled
    error_std = np.std(residuals)  # Key metric for confidence interval

    # --- Predict future values ---
    last_sequence = scaled_prices[-n_steps:]
    predicted = []
    current_sequence = last_sequence.copy()

    for _ in range(n_future_days):
        pred = model.predict(current_sequence.reshape(1, n_steps, 1), verbose=0)
        predicted.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], [[pred[0, 0]]], axis=0)

    # Rescale predictions
    predicted_prices = scaler.inverse_transform(
        np.array(predicted).reshape(-1, 1)
    ).flatten()
    future_dates = pd.date_range(
        df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_future_days
    )
    prediction_df = pd.DataFrame(
        {"Date": future_dates, "Predicted Price": predicted_prices}
    )

    # --- Plotting with confidence interval ---
    plt.rcParams["font.family"] = "Times New Roman "

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="none")
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)

    # Historical data
    ax.plot(df["Date"], df["Close"], label="Historical", color="cyan", linewidth=2)

    # Confidence interval (expanding uncertainty)
    days = np.arange(1, n_future_days + 1)
    expanding_std = error_std * np.sqrt(days)
    upper = predicted_prices + 1.96 * expanding_std  # 95% CI
    lower = predicted_prices - 1.96 * expanding_std

    ax.fill_between(
        prediction_df["Date"],
        lower,
        upper,
        color="lightblue",
        alpha=0.3,
        label="95% Confidence Interval",
    )

    # Predicted points (magenta dots)
    ax.plot(
        prediction_df["Date"],
        prediction_df["Predicted Price"],
        label=f"Next {n_future_days} Days Forecast",
        color="magenta",
        linestyle="None",
        marker=".",
        markersize=5,
    )

    # ---- NEW: Trend line spanning historical + forecasted data ----
    # Combine historical and predicted dates/prices
    all_dates = np.concatenate([df["Date"].values, prediction_df["Date"].values])
    all_prices = np.concatenate(
        [df["Close"].values, prediction_df["Predicted Price"].values]
    )

    # Calculate rolling mean (smoothing)
    window_size = 30  # Adjust based on your data frequency
    trend_line = pd.Series(all_prices).rolling(window=window_size, min_periods=1).mean()

    # Plot the trend line (blue dashed)
    ax.plot(
        all_dates,
        trend_line,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label="Long-Term Trend",
    )

    # Style
    ax.set_title(
        f"📈 Stock Price Forecast ({st.session_state.model})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig, prediction_df


#####################################################################################################

# Function to load data


# Load the data
# data = load_data()
# import matplotlib.pyplot as plt
# Path to your logo image
encoded_logo = "tensorflow.png"
main_bg_ext = "png"
main_bg = "Picture3.png "


if st.session_state.framework == "lstm":
    bg_color = "#FF5733"  # For example, a warm red/orange
    bg_color_iv = "orange"  # For example, a warm red/orange
    text_h1 = "BI-DIRECTIONAL"
    text_i = "Long short term memory"
    model = "TENSORFLOW"
    st.session_state.model = "best_bilstm_model.h5"
if st.session_state.framework == "gru":
    bg_color = "#FF5733"  # For example, a warm red/orange
    bg_color_iv = "orange"  # For example, a warm red/orange
    text_h1 = "GATED RECURRENT UNIT"
    text_i = "Recurrent Neural Network"
    model = "TENSORFLOW"
    st.session_state.model = "best_gru_model.h5"
if st.session_state.framework == "gen":
    bg_color = "#FF5733"  # For example, a warm red/orange
    bg_color_iv = "orange"  # For example, a warm red/orange
    text_h1 = "Amazon Stock Predictor"
    text_i = "21 Days Ahead of the Market"
    model = "TENSORFLOW"
st.markdown(
    f"""
    <style>
       /* Container for logo and text */
        /* Container for logo and text */
        .logo-text-container {{
            position: fixed;
            top: 20px; /* Adjust vertical position */
            left: 30px; /* Align with sidebar */
            display: flex;
            align-items: center;
            gap: 25px;
            width: 70%;
            z-index:1000;
        }}

        /* Logo styling */
        .logo-text-container img {{
            width: 50px; /* Adjust logo size */
            border-radius: 10px; /* Optional: round edges */
            margin-left:-5px;
            margin-top: -15px;

        }}

        /* Bold text styling */
        .logo-text-container h1 {{
            font-family: Nunito;
            color: #0175C2;            
            font-size: 25px;
            font-weight: bold;
            margin-right :100px;
            padding:0px;
            top:0;
            margin-top: -12px;
        }}
         .logo-text-container i{{
             font-family: Nunito;
            color: orange;            
            font-size: 15px;
            margin-right :10px;
            padding:0px;
            margin-left:-18.5%;
            margin-top:1%;
         }}

        /* Sidebar styling */
        section[data-testid="stSidebar"][aria-expanded="true"] {{
            margin-top: 100px !important; /* Space for the logo */
            border-radius: 0 60px 0px 60px !important; /* Top-left and bottom-right corners */
            width: 200px !important; /* Sidebar width */
            background: none; /* No background */
            color: white !important;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
            margin-right: 100px !important;
            margin-top: 1px !important;
            z-index: 1 !important;
            
            color: blue; /* White text */
            font-family:  "Times New Roman " !important; /* Font */
            font-size: 18px !important; /* Font size */
            font-weight: bold !important; /* Bold text */
            padding: 10px 20px; /* Padding for buttons */
            border: none; /* Remove border */
            border-radius: 35px; /* Rounded corners */
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); /* Shadow effect */
            transition: all 0.3s ease-in-out; /* Smooth transition */
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 10px 0;
            width:90%;
            left:5.5%;
            height:60px;
            margin-top:70px;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.4); /* Light border */

        }}

        div[data-testid="stDecoration"] {{
            background-image: none;
        }}

        div[data-testid="stApp"] {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;  /* Ensure the image covers the full page */
            background-position: center;
            background-repeat:no-repeat;
            height: 98vh;
            width: 99.3%;
            border-radius: 20px !important;
            margin-left: 5px;
            margin-right: 20px;
            margin-top: 10px;
            overflow: hidden;
            backdrop-filter: blur(10px); /* Glass effect */
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2); /* Light border */

        }}

        div[data-testid="stSidebarNav"] {{
            display: none;
        }}
        
        div[data-testid="stSlider"] {{
            margin-top:45px;
        }}
        label[data-testid="stWidgetLabel"]{{
            margin-left:20px !important;
        }}
        
        div[data-baseweb="slider"] {{
            border-radius: 30px;
            padding-right:40px;
            z-index: 1;
           /* Glass effect background */
            backdrop-filter: blur(2px);
            -webkit-backdrop-filter: blur(12px);
            /* Shiny blue borders (left & right) */
            border-top: 2px solid rgba(255, 255, 155, 0.4); /* Light border */
            margin-left:13px;
            border-bottom: 2px solid rgba(255, 255, 155, 0.4); /* Light border */
            
            
            }}
               div[data-baseweb="slider"] > :nth-child(1)> div {{
               margin-left:20px !important;
               margin-top:10px;
               }}
                div[data-testid="stSliderTickBarMin"]{{
                    background:none !important;
                    margin-left:20px !important;
                    font-size:12px;
                    margin-bottom:5px;
                    font-family:  "Times New Roman " !important; /* Font */
                }}
                 div[data-testid="stSliderTickBarMax"]{{
                    background:none !important;
                    font-size:12px;
                    margin-bottom:5px;

                    font-family:  "Times New Roman " !important; /* Font */
                }}
                div[data-testid="stSliderThumbValue"]{{
                    font-size:12px;
                    font-family:  "Times New Roman " !important; /* Font */

                }}
                div[data-testid="stProgress"]{{
                    margin-right:25px;
                    margin-top:-70px;
                    height:10px !important;
                    
                }}
          [class*="st-key-content-container-3"] {{
            margin-top: 80px;  /* Adjust top margin */
            marging-left:50px !important;
            color:orange;

        }}
       
        /* Button row styling */
        .button-row {{
            display: flex;
            justify-content: flex-start;
            gap: 20px;
            margin-bottom: 20px;
        }}

       

        .custom-button:hover {{
            background-color: #0056b3;
        }}
        div.stButton > button p{{   
        color: orange !important;
        font-weight:bold;
        }}
        div.stButton > button {{                
        background: rgba(255, 255, 255, 0.2);
        color: orange !important; /* White text */
        font-family:  "Times New Roman " !important; /* Font */
        font-size: 18px !important; /* Font size */
        font-weight: bold !important; /* Bold text */
        padding: 10px 20px; /* Padding for buttons */
        border: none; /* Remove border */
        border-radius: 35px; /* Rounded corners */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); /* Shadow effect */
        transition: all 0.3s ease-in-out; /* Smooth transition */
         display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px 0;
        width:160px;
        height:50px;
        margin-top:5px;

    }}

    /* Hover effect */
    div.stButton > button:hover {{
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.4); /* Enhanced shadow on hover */
        transform: scale(1.05); /* Slightly enlarge button */
        transform: scale(1.1); /* Slight zoom on hover */
        box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.4); /* Glow effect */
    }}
    
         div[data-testid="stMarkdownContainer"] p {{
            font-family: "Times New Roman" !important; /* Elegant font for title */
            color:black !important;
            
        }}
   .titles{{
      margin-top:-50px !important;
      margin-left:-40px;
      font-family: "Times New Roman" !important; 

  }}
     .header {{
            display: flex;
            align-items: center;
            gap: 20px;  /* Optional: adds space between image and text */
        }}
        .header img {{
            height: 120px;  /* Adjust as needed */
            margin-top:-15px;
        }}
    /* Title styling */
.header h1{{
   font-family: "Times New Roman" !important; /* Elegant font for title 
    font-size: 2.7rem;
    font-weight: bold;
    margin-left: 5px;
   /* margin-top:-50px;*/
    margin-bottom:30px;
    padding: 0;
    color: black; /* Neutral color for text */
    }}
 .titles .content{{
   font-family: "Times New Roman" !important; /* Elegant font for title */
    font-size: 1.2rem;
    margin-left: 150px;
    margin-bottom:1px;
    padding: 0;
    color:black; /* Neutral color for text */
    }}
  



    </style>
  
    """,
    unsafe_allow_html=True,
)
# Overlay container
st.markdown(
    f"""
    <style>
        .logo-text-containers {{
            position: absolute;
            top: -60px;
            right: 40px;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            z-index: 10;
            width:80vw;
            height:620px;
        }}
        .logo-text-containers img {{
            height: 40px;
           right:0;
        }}
        .logo-text-containers h1 {{
            display: inline;
            font-size: 20px;
            vertical-align: middle;
        }}
        .logo-text-containers i {{
            display: block;
            margin-top: 5px;
            font-size: 14px;
            color: #333;
        }}
        
     [class*="st-key-close-btn"] {{
            top: 5px;
            font-size: 20px;
            font-weight: bold !important;
            cursor: pointer;
            position:absolute;
            margin-left:1150px;
            color:red !important;
            z-index:1000;
        }}
       [class*="st-key-close-btn"]:hover {{
            color: #f00;
        }}
    [class*="st-key-divider-col"] {{
            border-left: 3px solid rgba(255, 255, 155, 0.5); /* Light border */
            border-radius: 35px; /* Rounded corners */
            margin-top:-15px;
            margin-left:3px;

        }}
          [class*="st-key-col1"] {{
            border-right: 3px solid rgba(255, 255, 155, 0.5); /* Light border */
            border-radius: 35px; /* Rounded corners */
            margin-left:20px;
            margin-top:-15px;

        }}

    [class*="st-key-logo-text-containers"] {{
          margin: 10px; /* Set a margin inside the container */
        max-width: 100%;
        overflow: hidden;

            position: absolute;
           top:-43px;
           left:10px;
           overflow: hidden;
            background-color: tansparent;
            padding: 15px;
            border-radius: 30px;
            padding-right:40px;
            z-index: 1;
            width:88vw;
            height:660px;
           /* Glass effect background */
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            /* Shiny blue borders (left & right) */
            border-left: 3px solid rgba(255, 255, 255, 0.9); /* Light border */
            border-right: 3px solid rgba(255, 255, 255, 0.9); /* Light border */


                }}
    
        @media (max-width: 768px) {{
    .logo-text-container h1 {{
        font-size: 12px;
        
    }}
   .logo-text-container i {{
        font-size: 10px;
        ma
    }}
   
     
      .logo-text-container img {{
            width: 30px; /* Adjust logo size */
            border-radius: 10px; /* Optional: round edges */
            margin-left:15px;
            margin-top: -35px;

        }}
        
}}
    </style>
""",
    unsafe_allow_html=True,
)

if st.session_state.show_overlay:

    with st.container(key="logo-text-containers"):
        if st.button("✕", key="close-btn"):
            st.session_state.show_overlay = False
            st.session_state.framework = "gen"
            st.rerun()
        with st.spinner("Downloading and processing the Data..."):
            progress_bar = st.progress(0)
            for i in range(1, 11):
                time.sleep(0.6)
                progress_bar.progress(i * 10)
        with st.container(key="content"):
            days = st.slider(
                "Amazon Stock Insight: Predictive Analytics Over 21 Days",
                1,
                21,
                7,
                key="days_slider",
            )

            col1, col2 = st.columns([2.5, 3])
            data = load_data()
            if data is not None and not data.empty:
                fig, future_data = predict_future_prices(
                    data, days+1, st.session_state.model
                    )
                with col1:
                    with st.container(key="col1"):
                        future_data["Date"] = future_data["Date"].dt.strftime("%Y-%m-%d")
                        future_data = future_data[1:]
                        styled_html = (
                        future_data.style.set_table_attributes('class="glass-table"')
                        .set_table_styles(
                            [
                                {
                                    "selector": "th",
                                    "props": [
                                        ("padding", "12px"),
                                        ("color", "#000"),
                                        (
                                            "background-color",
                                            "rgba(255, 255, 255, 0.15)",
                                        ),
                                    ],
                                },
                                {
                                    "selector": "td",
                                    "props": [
                                        ("padding", "10px"),
                                        ("color", "#000"),
                                        ("border-bottom", "1px solid rgba(0,0,0,0.1)"),
                                    ],
                                },
                                {
                                    "selector": "table",
                                    "props": [
                                        ("width", "100%"),
                                        ("border-collapse", "collapse"),
                                    ],
                                },
                            ]
                        )
                        .to_html()
                    )

                   
                        # Glassmorphism CSS + vertical scroll + black text
                        glass_css = """
                            <style>
                            /* Outer shell for glass effect & border radius */
                            .outer-glass-wrapper {
                                backdrop-filter: blur(10px);
                                -webkit-backdrop-filter: blur(10px);
                                background: rgba(255, 255, 255, 0.15);
                                border-radius: 20px;
                                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
                                max-height: 600px;
                                max-width: 800px;
                                overflow: hidden;
                                margin-right: 15px;
                                margin-left:3px;
                                font-family:  "Times New Roman " !important; /* Font */

                                font-size: 14px;
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                margin-bottom:30px;

                            }

                            /* Inner scrolling container */
                            .glass-container {
                                max-height: 410px;
                                overflow-y: auto;
                                padding: 16px 24px 16px 16px; /* right padding gives room for scrollbar */
                            }

                            /* Scrollbar styles */
                            .glass-container::-webkit-scrollbar {
                                width: 4px;
                            }
                            .glass-container::-webkit-scrollbar-track {
                                background: transparent;
                            }
                            .glass-container::-webkit-scrollbar-thumb {
                                background-color: rgba(0, 0, 0, 0.3);
                                border-radius: 10px;
                            }
                            .glass-container {
                                scrollbar-width: thin;
                                scrollbar-color: rgba(0, 0, 0, 0.3) transparent;
                            }

                            /* Table styling */
                            .glass-table {
                                width: 100%;
                            }
                            .glass-table th, .glass-table td {
                                text-align: left;
                                white-space: nowrap;
                                color: #000;
                            }
                            </style>
                            """

                        st.markdown(glass_css, unsafe_allow_html=True)
                        st.markdown(
                            f""" <div class="outer-glass-wrapper">
                        <div class="glass-container">
                            {styled_html}</div> </div>
                        """,
                            unsafe_allow_html=True,
                        )

                with col2:
                    with st.container(key="divider-col"):
                        st.pyplot(fig)

            else:
                st.error("No data loaded. Please check your internet connection.")
# Show overlay if triggered
st.markdown(
    f""" <div class="logo-text-container">
    <img src="data:image/png;base64,{base64.b64encode(open("tensorflow.png","rb").read()).decode()}" alt="Uploaded Image">
<h1>{text_h1}<br></h1>
<i>{  text_i}</i>
</div>

""",
    unsafe_allow_html=True,
)


st.markdown(
    f""" <div class="titles">
            <div class = "header">
            <img src="data:image/png;base64,{base64.b64encode(open("logo2.png","rb").read()).decode()}" alt="Uploaded Image">
            <h1></br>ACTIONS </br> TREND ANALYTICS</h1>
            </div>
            <div class="content">
            A deep learning-powered tool that analyzes Amazon's stock trends.<br>
            the models(BI-Direcional Lstm and GRU) predicts future market<br> actions based on past trends,
            providing a confidence score to <br> help users interpret the data more accurately and take timely actions. 
                </div>
                </div>
                        """,
    unsafe_allow_html=True,
)


with st.container(key="content-container-3"):
    col1, col2 = st.columns([1.5, 10.5])
    with col1:
        if st.button(" BIDIR-LSTM"):
            st.session_state.framework = "lstm"
            st.session_state.show_overlay = True
            st.rerun()
    with col2:
        if st.button("GRU"):
            st.session_state.framework = "gru"
            st.session_state.show_overlay = True
            st.rerun()
