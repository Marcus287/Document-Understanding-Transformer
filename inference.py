from donut import DonutModel
from PIL import Image

# Load pretrained model
model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

# Load image
image = Image.open("sample_invoice.png").convert("RGB")

# Define a sample prompt
prompt = "<s_docvqa><s_question>What is the invoice number?</s_question><s_answer>"

# Run inference
result = model.inference(image, prompt=prompt)
print("Output:", result)