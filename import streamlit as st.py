import streamlit as st
from PIL import Image
from donut import DonutModel

model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

st.title("Document Understanding with Donut")

uploaded_file = st.file_uploader("Upload a form image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Document", use_column_width=True)

    if st.button("Analyze"):
        prompt = "<s_docvqa><s_question>What is the invoice number?</s_question><s_answer>"
        result = model.inference(image, prompt=prompt)
        st.write("Answer:", result)