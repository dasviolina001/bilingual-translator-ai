# Bilingual Translator AI 🌐🤖

An AI-powered language translation system that translates English into **Assamese** and **Bodo** using deep learning models (Seq2Seq with Attention and Transformer). The system includes a virtual robot interface for interactive translation via a web-based UI.

## 🧠 Features

- 🔤 Translate English ➝ Assamese  
- 🔤 Translate English ➝ Bodo  
- 🧠 Models: Seq2Seq with Attention and Transformer  
- 📈 BLEU score evaluation and early stopping for best model accuracy  
- 🧩 Subword tokenization using SentencePiece  
- 🌐 Interactive web UI with a virtual robot character  
- 🔊 Text-to-Speech functionality (Speak button)  

## 🗂️ Project Structure

```
project/
├── backend/
│   ├── app.py
│   ├── translate_assamese.py
│   ├── translate_bodo.py
│   ├── asmm_engg_cleaned.csv
│   ├── csv_bodo_eng.csv
│   └── models/
│       ├── encoder.pth
│       ├── decoder.pth
│       ├── encoder_bodo.pth
│       └── decoder_bodo.pth
└── ui/
    └── index.html
```

### 🔗 Model Files

Due to file size limits on GitHub, all trained model files are available via Google Drive:

📁 [Download Trained Models (encoder/decoder .pth files)](https://drive.google.com/drive/folders/1ruo0E7GBQ6thgSCgtSlWHZ7xOHuFUusF?usp=drive_link)

Make sure to:
- Download all files from the link above
- Place them inside a folder named `models` in your project root directory


## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bilingual-translator-ai.git
cd bilingual-translator-ai
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Start the backend (Flask API)
```bash
cd backend
python app.py
```

### 4. Open the UI
Open `ui/index.html` in your browser.

## 📸 Screenshots

### 🔘 Main UI

<img width="839" height="855" alt="Screenshot (77)" src="https://github.com/user-attachments/assets/0042da90-5163-49a3-9186-64de0e2b0ef4" />


## 🛠️ Technologies Used

- Python (PyTorch, Flask)  
- SentencePiece  
- HTML, CSS, JavaScript (for UI)  
- Framer Motion (animations)  
- Text-to-Speech API (for speak feature)  

## 📊 Model Highlights

- **Assamese Translation**: Trained with attention-based Seq2Seq model  
- **Bodo Translation**: Trained with a similar architecture, fine-tuned separately  
- **Transformer models**: Also tested and implemented for both languages  
- **Evaluation**: BLEU Score, training/validation loss tracking  

## 🌟 Future Enhancements

- Add Assamese/Bodo ➝ English reverse translation  
- Voice input for English  
- Deploy on Hugging Face Spaces or Render  
- Extend to more regional Indian languages  

## 👩‍💻 Developed By

Violina Das  
B.Tech in Computer Science & Engineering  


📌 *This project was built as part of an academic and personal initiative to preserve and promote low-resource languages using AI.*
