import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LEN    = 128
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading model...')
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2)
try:
    model.load_state_dict(torch.load('best_mbert.pt', map_location=DEVICE))
    print('Loaded fine-tuned weights')
except:
    print('Using base weights (run training notebook first)')
model = model.to(DEVICE).eval()

EXAMPLES = [
    'BREAKING! Government HIDING cure for cancer! Big Pharma CONSPIRACY! SHARE before deleted!',
    'WHO published new guidelines on vaccine schedules after review by 50 global experts.',
    'அதிர்ச்சி! பிரபல நடிகர் இரகசியமாக வெளிநாட்டிற்கு பணம் அனுப்பினார்! உடனே பகிருங்கள்!',
    'தமிழ்நாட்டில் புதிய சுகாதாரக் கொள்கை அமல்படுத்தப்படும் என்று அமைச்சகம் அறிவித்தது.',
]

def classify(text):
    if not text or not text.strip():
        return 'Please enter some text.', {}
    enc = tokenizer(text, max_length=MAX_LEN, truncation=True,
                    padding='max_length', return_tensors='pt')
    with torch.no_grad():
        out   = model(input_ids=enc['input_ids'].to(DEVICE),
                      attention_mask=enc['attention_mask'].to(DEVICE))
        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
    real_p, fake_p = float(probs[0]), float(probs[1])
    verdict = 'FAKE NEWS' if fake_p > 0.5 else 'REAL NEWS'
    emoji   = 'FAKE NEWS' if fake_p > 0.5 else 'REAL NEWS'
    detail  = f'Confidence: {max(real_p,fake_p):.1%} | Language auto-detected'
    return f'{emoji} — {verdict}\n{detail}', {'REAL': real_p, 'FAKE': fake_p}

with gr.Blocks(title='Fake News Detector', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# Fake News Detector — Tamil & English')
    gr.Markdown('Powered by **mBERT** fine-tuned on bilingual fake news data')
    with gr.Row():
        with gr.Column():
            txt  = gr.Textbox(label='News Article (Tamil or English)',
                              placeholder='Paste your news article here...',
                              lines=6)
            btn  = gr.Button('Classify', variant='primary')
            gr.Examples(examples=EXAMPLES, inputs=txt)
        with gr.Column():
            out_label = gr.Textbox(label='Verdict')
            out_conf  = gr.Label(label='Confidence Scores')
    btn.click(classify, inputs=txt, outputs=[out_label, out_conf])
    gr.Markdown('*Stage 11 Portfolio Project — mBERT Fine-Tuning*')

if __name__ == '__main__':
    demo.launch(share=False)