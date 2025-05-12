import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

@st.cache_resource
def load_models():
    # Summarizer model (e.g., BART)
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # Text classifier model (e.g., BERT for sentiment)
    classifier_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    classifier = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

    # Reply generator model (e.g., DialoGPT)
    replier_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    replier = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    return summarizer, summarizer_tokenizer, classifier, classifier_tokenizer, replier, replier_tokenizer

# Load models
summarizer, summarizer_tokenizer, classifier, classifier_tokenizer, replier, replier_tokenizer = load_models()

# Streamlit App
st.title("AI Assistant - Summarize, Classify & Generate Replies")
st.write("Project by Arshad Anwar")

# Sidebar selection
task = st.sidebar.selectbox("Choose Task", ["Summarize", "Classify", "Generate Reply"])

# Input
user_input = st.text_area("Enter email text here:")

# Run button
if st.button("Run"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        if task == "Summarize":
            inputs = summarizer_tokenizer.encode(user_input, return_tensors="pt", max_length=384, truncation=True)
            summary_ids = summarizer.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
            summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.success("📝 Summary:")
            st.write(summary)

        elif task == "Classify":
            inputs = classifier_tokenizer(user_input, return_tensors="pt", max_length=384, truncation=True, padding=True)
            outputs = classifier(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label_id = torch.argmax(probs).item()

            # Updated classification labels
            labels = ['Work', 'Personal', 'Spam', 'Finance', 'Legal', 'HR']

            # Ensure the number of labels matches your model's output size
            if len(labels) != probs.shape[1]:
                st.error("⚠️ The number of labels doesn't match the model's output dimension.")
            else:
                st.success("📌 Prediction:")
                st.write(f"{labels[label_id]} ({probs[0][label_id]*100:.2f}%)")

        elif task == "Generate Reply":
            input_ids = replier_tokenizer.encode(user_input + replier_tokenizer.eos_token, return_tensors="pt")
            output = replier.generate(input_ids, max_length=100, pad_token_id=replier_tokenizer.eos_token_id)
            reply = replier_tokenizer.decode(output[0], skip_special_tokens=True)
            st.success("✉️ Suggested Reply:")
            st.write(reply)
