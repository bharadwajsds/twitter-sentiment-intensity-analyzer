import streamlit as st
import base64

# Page title and description
st.title(" INDUSTRIAL ORIENTED MINI PROJECT (20261A0552)")
st.title("Sentiment Analysis & Emotion Classification Application")
st.write("Welcome to our sentiment analysis and emotion classification app! Click the button below to proceed to the analysis.")

# Add the URL or path to your background image
background_image = "C:/Users/bhara/OneDrive/Desktop/MINI PROJECT/abstract-flowing-neon-wave-background_53876-101942 (2).jpg" 

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background(background_image)

# Set the background image using CSS
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url("{background_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Redirect button to the second Streamlit app
if st.button("Start Analysis"):
    # Redirects to the second Streamlit app using a URL
    st.markdown('[Click here to begin analysis](http://localhost:8502/)', unsafe_allow_html=True)

# Additional section for information or instructions
st.header("How to use the app:")
st.write("""
- Enter text in the provided fields for sentiment and emotion analysis.
- Translate text into various languages for analysis.
- Upload a CSV file to analyze sentiments across a dataset.
""")

# Additional section for contact or additional information
st.header("Contact Us:")
st.write("For support or inquiries, please reach out to us at bharadwajsds@gmail.com")
