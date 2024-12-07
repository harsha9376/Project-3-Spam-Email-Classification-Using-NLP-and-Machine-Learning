import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

# Define the main function for the app
def main():
    # Title and description of the app
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or not spam.")
    
    # Subheader for the classification section
    st.subheader("Classification")
    
    # User input field for entering an email
    user_input = st.text_area("Enter an email to classify", height=200)
    
    # When the "Classify" button is pressed
    if st.button("Classify"):
        if user_input:  # If user has entered some text
            # Preprocess the input and transform it using the vectorizer
            data = [user_input]
            vec = cv.transform(data).toarray()
            
            # Predict the result using the trained model
            result = model.predict(vec)
            
            # Display the result
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            # If no input is entered, ask the user to input an email
            st.write("Please enter an email to classify.")

# Run the main function to start the app
if __name__ == "__main__":
    main()
