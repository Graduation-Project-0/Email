#!/usr/bin/env python
# coding: utf-8

# In[6]:
from fastapi import FastAPI, File, UploadFile
import joblib
import re
import string
import base64
from nltk.corpus import stopwords
import os

stop_words = set(stopwords.words('english'))

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="Email Spam Detection")

#------------------------------------------------------------------------------------------------------------
def extract_email(raw_email):
     email_text = re.findall(
            r"Content-Type:\s*text/plain;[^\n\r]*[\r\n]+Content-Transfer-Encoding:\s*[^\n\r]+[\r\n]+([\s\S]*?)(?=\n--|\r--|$)",
            raw_email,
            re.DOTALL | re.IGNORECASE
        )
     extracted_text = ""
     for row in email_text:
        row = row.strip()
        try:
            decoded_text = base64.b64decode(row.strip()).decode("utf-8", errors="ignore")
        except Exception:
            decoded_text = row
        extracted_text += decoded_text + "\n"
     return extracted_text.strip()
#-------------------------------------------------------------------------------------------------------------

def extract_url(extracted_text):
    urls = re.findall(r"(https?://[^\s]+|www\.[^\s]+)", extracted_text)
    return urls
#---------------------------------------------------------------------------------------------------------------------
def extract_images(raw_email, save_dir ="attachments/extracted_images"):
    os.makedirs(save_dir, exist_ok=True)
    image_all_info = re.findall(
            r"Content-Type:\s*image\/[^\n\r]+[\s\S]*?(?=\n--|\r--|$)",
            raw_email,
            re.DOTALL | re.IGNORECASE,
        )
    extracted_img = []
    for i, part in enumerate(image_all_info, start= 1):
        split_image = re.search(r"\r?\n\s*\r?\n", part)
        if not split_image:
            continue
        split_index = split_image.start()
        header = part[:split_index].strip()
        encoded_image = part[split_image.end():].strip()
        img_name = re.search(r'filename=["\']([^"\']+)["\']', header, re.IGNORECASE)
        if img_name:
            image_name = img_name.group(1)
        else:
            image_name = f"image{i}.png"
        encoded_image = re.sub(r'\s+', '', encoded_image)
        image_bytes = base64.b64decode(encoded_image)
        file_path = os.path.join(save_dir, image_name)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        extracted_img.append({
            "filename": image_name,
            "encoded_image": encoded_image,
            "path": str(file_path)
            })
    return extracted_img  
#-----------------------------------------------------------------------------------------------------
def extract_files(raw_email, save_dir ="attachments/extracted_files"):
    os.makedirs(save_dir, exist_ok=True)
    file_all_info = re.findall(
            r"Content-Type:\s*application\/[^\n\r]+[\s\S]*?(?=\r?\n--|\r?--|$)",
            raw_email,
            re.DOTALL | re.IGNORECASE,
        )
    extracted_files = []
    for i, part in enumerate(file_all_info, start= 1):
        split_file = re.search(r"\r?\n\s*\r?\n", part)
        if not split_file:
            continue
        split_index = split_file.start()
        header = part[:split_index].strip()
        encoded_file = part[split_file.end():].strip()
        file_name = re.search(r'filename=["\']([^"\']+)["\']', header, re.IGNORECASE)
        if file_name:
            fileName = file_name.group(1)
        else:
            fileName = f"file{i}"
        encoded_file = re.sub(r'\s+', '', encoded_file)
        file_bytes = base64.b64decode(encoded_file)
        file_path = os.path.join(save_dir, fileName)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        extracted_files.append({
            "filename": fileName,
            "encoded_file": encoded_file,
            "path": str(file_path)
            })
    return extracted_files

#-------------------------------------------------------------------------------------------------------------

def clean_email(extracted_text):
    extracted_text = extracted_text.lower()
    extracted_text = re.sub(r'http\S+|www.\S+', '', extracted_text)
    extracted_text = extracted_text.translate(str.maketrans('', '', string.punctuation))
    extracted_text = re.sub(r'\d+', '', extracted_text)
    words = extracted_text.split()
    words = [w for w in words if w not in stop_words]
    email_clean = " ".join(words)
    return email_clean
#-----------------------------------------------------------------------------------------------------------

@app.post("/predict")
async def predict_email(file: UploadFile = File(...)):
    raw_text = await file.read()
    raw_email = raw_text.decode("utf-8", errors= "ignore")
    extracted_text = extract_email(raw_email)

    urls = extract_url(extracted_text)

    images = extract_images(raw_email)
    files = extract_files(raw_email)

    cleaned = clean_email(extracted_text)

    x = vectorizer.transform([cleaned])
    prediction = model.predict(x)[0]
    if prediction == 1:
        result = "Spam" 
    else:
        result = "Not Spam"

    return {
        "prediction": result,
        "num_urls": len(urls),
        "urls": urls,
        "num_images": len(images),
        "images": [{"filename":img["filename"], "encoded_image":img["encoded_image"]} for img in images],
        "num_files": len(files),
        "files": [{"filename":file["filename"], "encoded_file":file["encoded_file"]} for file in files],
        "text_preview": cleaned[:300]
    }


# In[ ]:




