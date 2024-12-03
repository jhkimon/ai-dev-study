import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Hello, Streamlit!")

user_input = st.text_input("Enter the name", "Guest")

if st.button("Say GOOOD"):
    st.write(f"Hello, {user_input}")

slider_value = st.slider("Pick number", 0, 100)
st.write(f"slider_value {slider_value}")


if st.checkbox("checkbox"):
    st.write("checkout!")

df = pd.DataFrame({"A": [1,2,3]})
st.dataframe(df)

x = np.linspace(0, 100, 200)
y = np.sin(x)
plt.plot(x, y)
st.pyplot(plt)
