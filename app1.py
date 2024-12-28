import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Title and description
st.title("SMS Spam Detection")
st.markdown("""
This application uses a trained machine learning model to detect whether an SMS message is **Spam** or **Not Spam**.
Enter an SMS message below to classify it.
""")

# Input box for user SMS message
sms = st.text_area("Enter the SMS message:")

# Predict spam or not
if st.button("Classify"):
    if sms.strip():
        # Transform input using the vectorizer
        transformed_sms = vectorizer.transform([sms])
        prediction = model.predict(transformed_sms)[0]  # Get prediction (0 or 1)

        # Display the result
        if prediction == 1:
            st.error("The message is classified as **Spam**.")
        else:
            st.success("The message is classified as **Not Spam**.")
    else:
        st.warning("Please enter a valid SMS message.")

# Footer
st.markdown("""
---
*Created as part of a Machine Learning NLP Project.*
""")
