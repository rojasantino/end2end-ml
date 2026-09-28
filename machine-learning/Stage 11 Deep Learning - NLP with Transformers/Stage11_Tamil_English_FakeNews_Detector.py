#!/usr/bin/env python
# coding: utf-8

# # 📰 Stage 11: Deep Learning — NLP with Transformers
# ## Project: Tamil & English Fake News Detector

# ## 📦 1. Imports & Configuration

# In[18]:


import os, re, time, random, warnings, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY']  = 'error'

# ML
from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.metrics           import (accuracy_score, f1_score, roc_auc_score,
                                       classification_report, confusion_matrix,
                                       roc_curve, precision_recall_curve,
                                       average_precision_score)
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import LabelEncoder

# HuggingFace
import torch
from torch.utils.data          import Dataset, DataLoader
from torch.optim               import AdamW
from transformers              import (AutoTokenizer, AutoModelForSequenceClassification,
                                       get_linear_schedule_with_warmup,
                                       pipeline as hf_pipeline)
# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = 'bert-base-multilingual-cased'
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Plot theme ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#080810','axes.facecolor':'#10101e',
    'axes.edgecolor':'#35355a','text.color':'#dcdcff',
    'axes.labelcolor':'#dcdcff','xtick.color':'#8888b0',
    'ytick.color':'#8888b0','grid.color':'#202038','grid.alpha':0.6,
    'legend.facecolor':'#181830','legend.edgecolor':'#404060',
})
C1,C2,C3,C4 = '#00d4ff','#00ff9f','#ffd700','#ff6b6b'

print(f"Device      : {DEVICE}")
print(f"Model       : {MODEL_NAME}")
print(f"Transformers: {__import__('transformers').__version__}")
print(f"PyTorch     : {torch.__version__}")


# ## 📰 2. Synthetic Bilingual Fake News Dataset
# 
# We generate a realistic dataset with 4,000 news articles:
# - **2,000 English** (1,000 real · 1,000 fake)
# - **2,000 Tamil** (1,000 real · 1,000 fake)

# In[19]:


np.random.seed(42); random.seed(42)
N_PER_CLASS = 1000   # per language per class → 4000 total

# ── English corpus templates ──────────────────────────────────────────────────
EN_REAL_TEMPLATES = [
    "The {org} announced today that {policy} will take effect from {date}. "
    "Officials confirmed the decision after {days} days of deliberation. "
    "The measure is expected to {effect}, according to {source}.",

    "Researchers at {university} published findings in {journal} showing that {discovery}. "
    "The study, conducted over {years} years with {n} participants, concludes that {conclusion}. "
    "Peer reviewers praised the methodology.",

    "{city} reported {count} new cases of {disease} on {day}. "
    "Health authorities have deployed {resource} to manage the situation. "
    "Officials urge residents to {advice}.",

    "The {country} government approved a ${amount} billion budget for {sector}. "
    "The plan, reviewed by {committee}, aims to {goal} by {year}. "
    "Opposition parties offered mixed responses.",

    "Stock markets {direction} on {day} as {company} reported {metric} quarterly earnings. "
    "Analysts at {bank} revised their {year} forecast to {value}, citing {reason}.",
]

EN_FAKE_TEMPLATES = [
    "BREAKING: {celebrity} SECRETLY {scandal}! Government HIDING the TRUTH about {topic}! "
    "SHARE before they DELETE this! {source} confirms SHOCKING revelation that mainstream "
    "media REFUSES to cover. This changes EVERYTHING!",

    "EXCLUSIVE: Scientists DISCOVER that {food} CURES {disease} in just {days} days! "
    "Big Pharma DOESN'T want you to know! {count} doctors SILENCED for revealing this! "
    "WAKE UP! The {conspiracy} is REAL!",

    "ALERT: {politician} caught {crime} on camera — VIDEO PROOF! Deep state PANICKING! "
    "Over {count} million patriots DEMAND justice! {country} is FALLING because of {group}! "
    "RT if you AGREE! Share EVERYWHERE!",

    "LEAKED: {document} REVEALS that {organization} plans to {threat} by {year}! "
    "Insiders FLEE for their lives! {count}% of population at RISK! "
    "They are COMING for your {freedom}! RESIST now!",

    "MIRACLE: {person} LOSES {amount} kg in {days} days using ONE weird trick doctors HATE! "
    "Experts BAFFLED! {celebrity} uses this SECRET daily! "
    "ORDER before {date} — LIMITED SUPPLY!",
]

TA_REAL_TEMPLATES = [
    "{org} நிறுவனம் இன்று அறிவித்தது: {policy} {date} முதல் நடைமுறைக்கு வரும். "
    "{days} நாள் ஆலோசனைக்குப் பிறகு அதிகாரிகள் இந்த முடிவை உறுதிப்படுத்தினர். "
    "{source} தெரிவிக்கும் வகையில், இந்த நடவடிக்கை {effect} என்று எதிர்பார்க்கப்படுகிறது.",

    "{university}-இல் உள்ள ஆராய்ச்சியாளர்கள் {journal} இதழில் ஆய்வு முடிவுகளை வெளியிட்டனர். "
    "{years} ஆண்டுகள் {n} பங்கேற்பாளர்களுடன் நடத்தப்பட்ட இந்த ஆய்வு {discovery} என்று கண்டறிந்தது. "
    "சகாக்கள் ஆய்வு முறையை பாராட்டினர்.",

    "{city}-ல் {day} அன்று {count} புதிய {disease} வழக்குகள் பதிவாயின. "
    "சுகாதார அதிகாரிகள் நிலைமையை சமாளிக்க {resource} நிறுத்தினர். "
    "குடிமக்கள் {advice} என்று அறிவுறுத்தப்படுகின்றனர்.",

    "{country} அரசு {sector} துறைக்கு ₹{amount} கோடி நிதி ஒதுக்கியது. "
    "{committee} மதிப்பாய்வு செய்த திட்டம் {year}-ல் {goal} நோக்கமாக கொண்டுள்ளது. "
    "எதிர்க்கட்சிகள் கலவையான கருத்துக்களை தெரிவித்தன.",

    "{company} நிறுவனம் {metric} காலாண்டு வருவாய் அறிவித்ததால் {day} அன்று சந்தை {direction}. "
    "{bank} பகுப்பாய்வாளர்கள் {year} கணிப்பை {value} ஆக திருத்தினர்.",
]

TA_FAKE_TEMPLATES = [
    "அதிர்ச்சி செய்தி: {celebrity} இரகசியமாக {scandal}! அரசு {topic} பற்றிய உண்மையை மறைக்கிறது! "
    "DELETE ஆவதற்கு முன் SHARE செய்யுங்கள்! {source} இந்த அதிர்ச்சிகரமான உண்மையை உறுதிப்படுத்துகிறது!",

    "இரகசியம் வெளியாயிற்று: {food} {days} நாளில் {disease}-ஐ குணப்படுத்துகிறது! "
    "பெரிய நிறுவனங்கள் இதை மறைக்கின்றன! {count} மருத்துவர்கள் இதை வெளியிட்டதால் நிறுத்தப்பட்டனர்! "
    "விழிப்புணர்வு பெறுங்கள்! {conspiracy} உண்மையானது!",

    "அம்பலம்: {politician} கேமராவில் {crime} பிடிக்கப்பட்டனர்! {count} லட்சம் மக்கள் நீதி கோருகின்றனர்! "
    "{country} {group} காரணமாக நலிவடைகிறது! உடனே பகிருங்கள்!",

    "கசிந்த ஆவணம்: {organization} {year}-ல் {threat} திட்டமிடுகிறது! "
    "உள்நாட்டு தகவலாளர்கள் உயிருக்கு பயந்து ஓடுகின்றனர்! "
    "மக்களுக்கு {freedom} ஆபத்தில் உள்ளது! SHARE செய்யுங்கள்!",

    "அதிசயம்: {person} ஒரே {days} நாளில் {amount} கிலோ குறைத்தனர்! "
    "மருத்துவர்கள் வெறுக்கும் இந்த ஒரு இரகசிய உத்தி! "
    "{celebrity} தினமும் இதை பயன்படுத்துகிறார்! இன்றே ஆர்டர் செய்யுங்கள்!",
]

# ── Fill-in word banks ────────────────────────────────────────────────────────
FILLERS = dict(
    org        = ['WHO','UN','IMF','NASA','ISRO','RBI','SEBI','UNESCO'],
    policy     = ['new trade agreement','vaccine mandate','climate policy','tax reform'],
    date       = ['January 1','March 15','June 30','December 1','April 5'],
    days       = ['30','60','90','7','14','21'],
    effect     = ['reduce inflation','boost employment','improve healthcare','cut emissions'],
    source     = ['ministry spokesperson','senior official','Reuters','PTI','AFP'],
    university = ['MIT','IIT Madras','Oxford','Stanford','NUS','BITS Pilani'],
    journal    = ['Nature','Lancet','Science','NEJM','Cell'],
    discovery  = ['reduced risk by 40%','doubled lifespan','cut costs by 30%','improved accuracy'],
    years      = ['2','3','5','10'],
    n          = ['1,200','5,000','10,000','500'],
    conclusion = ['lifestyle changes matter','early screening saves lives','diet is key'],
    city       = ['Chennai','Mumbai','Delhi','Bengaluru','Kolkata','Hyderabad'],
    count      = ['12','45','100','2,300','500,000','10 million'],
    disease    = ['dengue','COVID-19','malaria','influenza','cholera'],
    day        = ['Monday','Tuesday','Wednesday','Thursday','Friday'],
    resource   = ['medical teams','vaccines','testing kits','mobile units'],
    advice     = ['stay home','get vaccinated','wear masks','avoid crowding'],
    country    = ['India','USA','UK','Germany','Japan','France'],
    amount     = ['5','12','50','200','1,000','2,500'],
    sector     = ['infrastructure','education','healthcare','defence','agriculture'],
    committee  = ['parliament','cabinet','expert panel','finance ministry'],
    goal       = ['achieve net-zero','eradicate poverty','boost GDP by 7%'],
    year       = ['2025','2026','2027','2030','2035'],
    direction  = ['rose','fell','surged','dipped','rallied','plunged'],
    company    = ['Reliance','TCS','Infosys','Apple','Tesla','Google','HDFC'],
    metric     = ['strong','record','disappointing','mixed','stellar'],
    bank       = ['Goldman Sachs','Morgan Stanley','ICICI','Citi','JPMorgan'],
    value      = ['7.5%','4.2%','8.1%','6.3%','9.0%'],
    reason     = ['strong demand','supply chain issues','cost pressures','innovation'],
    celebrity  = ['a major politician','a famous actor','a tech billionaire'],
    scandal    = ['betrays nation','funds terror groups','meets foreign spy'],
    topic      = ['5G towers','vaccines','water supply','chemtrails','climate'],
    conspiracy = ['New World Order','deep state plot','Big Pharma cover-up'],
    politician = ['a senior minister','the opposition leader','a governor'],
    crime      = ['accepting bribe','lying under oath','hiding documents'],
    group      = ['immigrants','elites','globalists','foreign agents'],
    document   = ['classified file','leaked email','secret memo'],
    organization = ['WHO','UN','WEF','government','deep state'],
    threat     = ['microchip citizens','control food supply','censor internet'],
    freedom    = ['freedom of speech','gun rights','privacy','property'],
    food       = ['turmeric','garlic','lemon water','coconut oil','neem'],
    person     = ['a mother of three','a retired teacher','a local farmer'],
)

def fill_template(template):
    result = template
    for key, options in FILLERS.items():
        placeholder = '{' + key + '}'
        while placeholder in result:
            result = result.replace(placeholder, random.choice(options), 1)
    return result

def make_articles(templates, n):
    articles = []
    for _ in range(n):
        t = random.choice(templates)
        articles.append(fill_template(t))
    return articles

print("Generating synthetic dataset ...")
t0 = time.time()

en_real = make_articles(EN_REAL_TEMPLATES, N_PER_CLASS)
en_fake = make_articles(EN_FAKE_TEMPLATES, N_PER_CLASS)
ta_real = make_articles(TA_REAL_TEMPLATES, N_PER_CLASS)
ta_fake = make_articles(TA_FAKE_TEMPLATES, N_PER_CLASS)

df = pd.DataFrame({
    'text'    : en_real + en_fake + ta_real + ta_fake,
    'label'   : ([0]*N_PER_CLASS + [1]*N_PER_CLASS)*2,
    'language': (['en']*N_PER_CLASS*2) + (['ta']*N_PER_CLASS*2),
})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['label_name'] = df['label'].map({0:'REAL', 1:'FAKE'})
df['text_len']   = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

df.to_csv('fakenews_bilingual.csv', index=False)
print(f"Done in {time.time()-t0:.1f}s")
print(f"Total articles : {len(df):,}")
print(f"Label balance  : {df['label_name'].value_counts().to_dict()}")
print(f"Language split : {df['language'].value_counts().to_dict()}")
print()
print(df[['text','label_name','language','word_count']].head(4).to_string(index=False))


# ## 🔍 3. Exploratory Data Analysis

# In[20]:


fig = plt.figure(figsize=(18, 12))
fig.suptitle('📊 Bilingual Fake News Dataset — EDA', fontsize=16, color=C1, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# Class balance
ax1 = fig.add_subplot(gs[0,0])
counts = df['label_name'].value_counts()
ax1.pie(counts, labels=counts.index, colors=[C2, C4],
        autopct='%1.1f%%', startangle=140,
        textprops={'color':'white','fontsize':11})
ax1.set_title('Class Distribution', color='#dcdcff')

# Language × class
ax2 = fig.add_subplot(gs[0,1])
cross = df.groupby(['language','label_name']).size().unstack()
cross.plot(kind='bar', ax=ax2, color=[C2,C4], edgecolor='white', linewidth=0.5, rot=0)
ax2.set_title('Language × Class', color='#dcdcff')
ax2.set_xlabel('Language'); ax2.set_ylabel('Count')
ax2.legend(['REAL','FAKE'], framealpha=0.3); ax2.grid(True, alpha=0.3, axis='y')

# Word count distribution
ax3 = fig.add_subplot(gs[0,2])
for lbl, color, name in [(0,C2,'REAL'), (1,C4,'FAKE')]:
    ax3.hist(df[df['label']==lbl]['word_count'], bins=30,
             alpha=0.65, color=color, label=name, density=True)
ax3.set_title('Word Count by Class', color='#dcdcff')
ax3.set_xlabel('Word Count'); ax3.legend(framealpha=0.3); ax3.grid(True, alpha=0.3)

# Top words in real vs fake
from sklearn.feature_extraction.text import CountVectorizer
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
for ax, lbl, color, title in [
    (ax4, 0, C2, 'Top 15 Words — REAL'),
    (ax5, 1, C4, 'Top 15 Words — FAKE'),
]:
    cv  = CountVectorizer(max_features=15, stop_words='english', min_df=2)
    sub = df[df['label']==lbl]['text'].str.lower()
    cv.fit(sub)
    freq = cv.transform(sub).toarray().sum(0)
    words = cv.get_feature_names_out()
    idx   = np.argsort(freq)
    ax.barh(words[idx], freq[idx], color=color, alpha=0.85,
            edgecolor='white', linewidth=0.4)
    ax.set_title(title, color='#dcdcff')
    ax.set_xlabel('Frequency'); ax.grid(True, alpha=0.3, axis='x')

# Text length by language
ax6 = fig.add_subplot(gs[1,2])
for lang, color, name in [('en',C1,'English'), ('ta',C3,'Tamil')]:
    ax6.hist(df[df['language']==lang]['text_len'], bins=30,
             alpha=0.65, color=color, label=name, density=True)
ax6.set_title('Text Length by Language', color='#dcdcff')
ax6.set_xlabel('Character Length'); ax6.legend(framealpha=0.3); ax6.grid(True, alpha=0.3)

plt.savefig('eda_fakenews.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ EDA plot saved → eda_fakenews.png")



# ## ⚙️ 4. Text Preprocessing & Splitting
# 

# In[ ]:


def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0B80-\u0BFF.,!?]', ' ', text)
    return text.strip()

df['text_clean'] = df['text'].apply(clean_text)

X = df['text_clean'].values
y = df['label'].values # ➡️ News Label : 0 = REAL; 1 = FAKE

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)


lang_test  = df['language'].values[df.index.isin(
    df.sample(frac=1, random_state=42).index[-len(X_test):])]

print(f"Train : {len(X_train):,}  ({y_train.mean():.1%} fake)")
print(f"Val   : {len(X_val):,}   ({y_val.mean():.1%} fake)")
print(f"Test  : {len(X_test):,}  ({y_test.mean():.1%} fake)")
print()
print("Sample REAL:", X_train[y_train==0][0][:120], "...")
print()
print("Sample FAKE:", X_train[y_train==1][0][:120], "...")


# ## 🧠 5. Transformer Architecture — Intuition
# 

# ## 🔤 6. Tokenisation — WordPiece Demo

# In[ ]:


print("Loading mBERT tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
print(f"Vocab size: {tokenizer.vocab_size:,}")

# ── Demo tokenisation ─────────────────────────────────────────────────────────
demo_texts = [
    "Climate change is a government hoax designed to control the population!",
    "Scientists published peer-reviewed evidence of rising temperatures in Nature journal.",
    "தமிழ்நாட்டில் புதிய கல்விக்கொள்கை அமல்படுத்தப்படும் என அரசு அறிவித்தது.",
    "அதிர்ச்சி! பிரபல நடிகர் இரகசியமாக வெளிநாட்டிற்கு பணம் அனுப்பினார்!",
]

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle('🔤 WordPiece Tokenisation — mBERT', fontsize=14, color=C1, fontweight='bold')

for ax, text in zip(axes.flat, demo_texts):
    enc    = tokenizer(text, max_length=32, truncation=True, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    ids    = enc['input_ids'][0].numpy()

    x_pos  = np.arange(len(tokens))
    is_sub = [t.startswith('##') for t in tokens]
    colors_tok = [C4 if s else (C3 if t in ['[CLS]','[SEP]'] else C1)
                  for t, s in zip(tokens, is_sub)]

    ax.bar(x_pos, [1]*len(tokens), color=colors_tok, edgecolor='#0a0a18',
           linewidth=0.8, width=0.85)
    for i, (tok, tid) in enumerate(zip(tokens, ids)):
        ax.text(i, 0.5, tok[:8], ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', rotation=65)

    lang = 'Tamil' if any(ord(c) > 0x0B7F for c in text) else 'English'
    ax.set_title(f'{lang}: "{text[:50]}..."', color='#dcdcff', fontsize=9)
    ax.set_ylim(0, 1.4); ax.axis('off')

from matplotlib.patches import Patch
legend_el = [Patch(color=C3, label='Special [CLS]/[SEP]'),
             Patch(color=C1, label='Full word'),
             Patch(color=C4, label='## Subword piece')]
fig.legend(handles=legend_el, loc='lower center', ncol=3,
           framealpha=0.3, fontsize=10)

plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig('tokenisation_demo.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Tokenisation demo saved → tokenisation_demo.png")
print(f"\nToken counts: {[len(tokenizer(t)['input_ids']) for t in demo_texts]}")


# ## 📊 7. Baseline Models — TF-IDF + Logistic Regression & SVM
# 

# In[ ]:


# ── TF-IDF pipeline ───────────────────────────────────────────────────────────
tfidf_params = dict(
    max_features = 50_000,
    ngram_range  = (1, 2),
    sublinear_tf = True,
    min_df       = 2,
    analyzer     = 'word',
)

pipelines = {
    'TF-IDF + LR' : Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]),
    'TF-IDF + SVM': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LinearSVC(C=0.5, max_iter=2000, random_state=42)),
    ]),
}

baseline_results = {}
for name, pipe in pipelines.items():
    t0 = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_prob = pipe.predict_proba(X_test)[:,1]
    except:
        y_prob = pipe.decision_function(X_test)
        y_prob = (y_prob - y_prob.min())/(y_prob.max()-y_prob.min())

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro')
    auc  = roc_auc_score(y_test, y_prob)
    elapsed = time.time()-t0

    baseline_results[name] = {'acc':acc,'f1':f1,'auc':auc,
                               'pred':y_pred,'prob':y_prob,'time':elapsed}
    print(f"{name:18s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  ({elapsed:.1f}s)")

# ── Feature importance (LR coefficients) ─────────────────────────────────────
lr_pipe    = pipelines['TF-IDF + LR']
feat_names = lr_pipe.named_steps['tfidf'].get_feature_names_out()
coefs      = lr_pipe.named_steps['clf'].coef_[0]
top_n      = 15
top_fake   = np.argsort(coefs)[-top_n:][::-1]
top_real   = np.argsort(coefs)[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('📊 TF-IDF LR — Top Discriminative Features', fontsize=13,
             color=C1, fontweight='bold')

for ax, indices, title, color in [
    (axes[0], top_fake, 'Top FAKE indicators',  C4),
    (axes[1], top_real, 'Top REAL indicators',  C2),
]:
    words = feat_names[indices]
    vals  = np.abs(coefs[indices])
    ax.barh(words[::-1], vals[::-1], color=color, alpha=0.85,
            edgecolor='white', linewidth=0.4)
    ax.set_title(title, color='#dcdcff')
    ax.set_xlabel('|Coefficient|')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('tfidf_features.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ TF-IDF features saved → tfidf_features.png")


# ## 🔧 8. PyTorch Dataset & DataLoader
# 

# In[ ]:


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length      = self.max_len,
            padding         = 'max_length',
            truncation      = True,
            return_tensors  = 'pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels'        : torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }

train_ds = FakeNewsDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = FakeNewsDataset(X_val,   y_val,   tokenizer, MAX_LEN)
test_ds  = FakeNewsDataset(X_test,  y_test,  tokenizer, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches : {len(train_dl)}")
print(f"Val   batches : {len(val_dl)}")
print(f"Test  batches : {len(test_dl)}")

# Inspect one batch
batch = next(iter(train_dl))
print(f"\nBatch keys : {list(batch.keys())}")
print(f"input_ids  : {batch['input_ids'].shape}")
print(f"attn_mask  : {batch['attention_mask'].shape}")
print(f"labels     : {batch['labels'][:8]}")


# ## 🤖 9. mBERT Model — Fine-Tuning Setup
# 

# In[ ]:


print(f"Loading {MODEL_NAME} ...")
t0 = time.time()
bert_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    ignore_mismatched_sizes=True,
)
bert_model = bert_model.to(DEVICE)
print(f"Loaded in {time.time()-t0:.1f}s")

total_params     = sum(p.numel() for p in bert_model.parameters())
trainable_params = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

# ── Optimizer & scheduler ────────────────────────────────────────────────────
optimizer = AdamW(bert_model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps,
)
print(f"\nOptimizer   : AdamW  LR={LR}  weight_decay=0.01")
print(f"Scheduler   : Linear warmup ({warmup_steps} steps) → {total_steps} total steps")
print(f"Epochs      : {EPOCHS}")
print(f"Batch size  : {BATCH_SIZE}")


# ## 🏋️ 10. Fine-Tuning Loop — mBERT
# 

# In[ ]:


def train_epoch(model, loader, optimizer, scheduler, device, clip=1.0):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids  = batch['input_ids'].to(device)
        attn_mask  = batch['attention_mask'].to(device)
        labels     = batch['labels'].to(device)

        out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds       = out.logits.argmax(dim=1)
        n_correct  += (preds == labels).sum().item()
        n_total    += labels.size(0)

    return total_loss / n_total, n_correct / n_total

def eval_epoch(model, loader, device):
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids  = batch['input_ids'].to(device)
            attn_mask  = batch['attention_mask'].to(device)
            labels     = batch['labels'].to(device)

            out   = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss  = out.loss
            probs = torch.softmax(out.logits, dim=1)[:,1]

            total_loss += loss.item() * labels.size(0)
            preds       = out.logits.argmax(dim=1)
            n_correct  += (preds == labels).sum().item()
            n_total    += labels.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss/n_total, n_correct/n_total, auc

# ── Training ──────────────────────────────────────────────────────────────────
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_auc':[]}
best_val_auc = 0.0
print(f"Fine-tuning mBERT for {EPOCHS} epochs on {DEVICE} ...")
print(f"Steps per epoch: {len(train_dl)}  |  Warmup: {warmup_steps}")
print("-"*65)

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(bert_model, train_dl, optimizer, scheduler, DEVICE)
    va_loss, va_acc, va_auc = eval_epoch(bert_model, val_dl, DEVICE)
    elapsed = time.time()-t0

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(va_loss)
    history['val_acc'].append(va_acc)
    history['val_auc'].append(va_auc)

    if va_auc > best_val_auc:
        best_val_auc = va_auc
        torch.save(bert_model.state_dict(), 'best_mbert.pt')
        saved = '✅ saved'
    else:
        saved = ''

    print(f"Epoch {epoch}/{EPOCHS}  "
          f"TrainLoss={tr_loss:.4f}  TrainAcc={tr_acc:.4f}  "
          f"ValLoss={va_loss:.4f}  ValAcc={va_acc:.4f}  "
          f"ValAUC={va_auc:.4f}  {elapsed:.0f}s  {saved}")

print(f"\nBest Val AUC: {best_val_auc:.4f}  → best_mbert.pt")


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('📈 mBERT Fine-Tuning — Training History', fontsize=14,
             color=C1, fontweight='bold')

epochs_range = range(1, EPOCHS+1)
for ax, (tr_key, va_key, title, ylabel) in zip(axes, [
    ('train_loss', 'val_loss', 'Loss',     'Cross-Entropy Loss'),
    ('train_acc',  'val_acc',  'Accuracy', 'Accuracy'),
    (None,         'val_auc',  'Val AUC',  'ROC-AUC'),
]):
    if tr_key:
        ax.plot(epochs_range, history[tr_key], color=C1, linewidth=2,
                marker='o', ms=7, label='Train')
    ax.plot(epochs_range, history[va_key], color=C3, linewidth=2,
            marker='s', ms=7, label='Val', linestyle='--')
    ax.set_title(title, color='#dcdcff')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.set_xticks(list(epochs_range))
    ax.legend(framealpha=0.3); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mbert_training.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Training curves saved → mbert_training.png")


# ## 📊 11. Test Set Evaluation

# In[ ]:


# Load best checkpoint
bert_model.load_state_dict(torch.load('best_mbert.pt', map_location=DEVICE))
bert_model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_dl:
        out   = bert_model(
            input_ids      = batch['input_ids'].to(DEVICE),
            attention_mask = batch['attention_mask'].to(DEVICE),
        )
        probs = torch.softmax(out.logits, dim=1)[:,1].cpu().numpy()
        preds = out.logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].numpy())

bert_preds = np.array(all_preds)
bert_probs = np.array(all_probs)
bert_labels = np.array(all_labels)

bert_acc = accuracy_score(bert_labels, bert_preds)
bert_f1  = f1_score(bert_labels, bert_preds, average='macro')
bert_auc = roc_auc_score(bert_labels, bert_probs)

print("="*60)
print(f"  mBERT (fine-tuned)  Acc={bert_acc:.4f}  F1={bert_f1:.4f}  AUC={bert_auc:.4f}")
print("="*60)
print()
print("Classification Report:")
print(classification_report(bert_labels, bert_preds,
                             target_names=['REAL','FAKE'], digits=4))


# ## 📈 12. ROC & Precision-Recall Curves — All Models

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('📈 Fake News Detector — ROC & PR Curves', fontsize=14,
             color=C1, fontweight='bold')

all_models = [
    ('TF-IDF + LR',  baseline_results['TF-IDF + LR']['prob'],  C3),
    ('TF-IDF + SVM', baseline_results['TF-IDF + SVM']['prob'], C2),
    ('mBERT',        bert_probs,                                C1),
]

# ROC
ax1 = axes[0]
ax1.plot([0,1],[0,1],'--',color='#606080',linewidth=1,label='Random (0.50)')
for name, prob, color in all_models:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax1.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name}  AUC={auc:.4f}')
ax1.fill_between(*roc_curve(y_test, bert_probs)[:2], alpha=0.08, color=C1)
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', color='#dcdcff')
ax1.legend(framealpha=0.3, fontsize=10); ax1.grid(True, alpha=0.3)

# PR
ax2 = axes[1]
baseline_pr = y_test.mean()
ax2.axhline(baseline_pr, linestyle='--', color='#606080', linewidth=1,
            label=f'Baseline ({baseline_pr:.2%})')
for name, prob, color in all_models:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax2.plot(rec, prec, color=color, linewidth=2.5, label=f'{name}  AP={ap:.4f}')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve', color='#dcdcff')
ax2.legend(framealpha=0.3, fontsize=10); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ ROC & PR curves saved → roc_pr.png")


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Confusion Matrices — All Models', fontsize=13, color=C1, fontweight='bold')

models_cm = [
    ('TF-IDF + LR',  baseline_results['TF-IDF + LR']['pred'],  C3),
    ('TF-IDF + SVM', baseline_results['TF-IDF + SVM']['pred'], C2),
    ('mBERT',        bert_preds,                                C1),
]
for ax, (name, preds, color) in zip(axes, models_cm):
    cm  = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                cbar=False, linewidths=0.5, linecolor='#080810',
                xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'])
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='macro')
    ax.set_title(f'{name}\nAcc={acc:.4f}  F1={f1:.4f}', color=color, fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Confusion matrices saved → confusion_matrices.png")


# ## 🌍 13. Language-Wise Performance Analysis

# In[ ]:


# Rebuild lang array aligned to test set
df_shuffled = pd.read_csv('fakenews_bilingual.csv')
_, df_test_idx = train_test_split(df_shuffled.index, test_size=0.15,
                                   random_state=42, stratify=df_shuffled['label'])
test_langs = df_shuffled.loc[df_test_idx, 'language'].values

# Evaluate per language
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('🌍 mBERT Performance by Language', fontsize=13, color=C1, fontweight='bold')

lang_metrics = {}
for lang, color, name in [('en', C1, 'English'), ('ta', C3, 'Tamil')]:
    mask = test_langs == lang
    if mask.sum() == 0:
        continue
    acc = accuracy_score(bert_labels[mask], bert_preds[mask])
    f1  = f1_score(bert_labels[mask], bert_preds[mask], average='macro')
    auc = roc_auc_score(bert_labels[mask], bert_probs[mask])
    lang_metrics[name] = {'Accuracy':acc, 'F1':f1, 'AUC':auc,
                           'preds':bert_preds[mask], 'labels':bert_labels[mask]}
    print(f"{name:8s}  n={mask.sum():4d}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

metric_names = ['Accuracy','F1','AUC']
x = np.arange(len(metric_names)); w = 0.35
for i, (lang_name, color) in enumerate([('English',C1),('Tamil',C3)]):
    if lang_name in lang_metrics:
        vals = [lang_metrics[lang_name][m] for m in metric_names]
        axes[0].bar(x+i*w, vals, w, label=lang_name, color=color,
                    alpha=0.85, edgecolor='white', lw=0.4)
axes[0].set_xticks(x+w/2); axes[0].set_xticklabels(metric_names)
axes[0].set_title('Metrics by Language', color='#dcdcff')
axes[0].set_ylim(0,1.15); axes[0].legend(framealpha=0.3); axes[0].grid(True,alpha=0.3,axis='y')

# ROC per language
for lang_name, color in [('English',C1),('Tamil',C3)]:
    if lang_name in lang_metrics:
        mask   = test_langs == ('en' if lang_name=='English' else 'ta')
        fpr,tpr,_ = roc_curve(bert_labels[mask], bert_probs[mask])
        axes[1].plot(fpr, tpr, color=color, linewidth=2,
                     label=f'{lang_name} AUC={lang_metrics[lang_name]["AUC"]:.4f}')
axes[1].plot([0,1],[0,1],'--',color='#606080',linewidth=1,label='Random')
axes[1].set_title('ROC by Language', color='#dcdcff')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(framealpha=0.3); axes[1].grid(True,alpha=0.3)

# Score distribution
for lang_name, color in [('English',C1),('Tamil',C3)]:
    if lang_name in lang_metrics:
        mask = test_langs == ('en' if lang_name=='English' else 'ta')
        axes[2].hist(bert_probs[mask & (bert_labels==0)], bins=25,
                     alpha=0.5, color=color, label=f'{lang_name} REAL', density=True)
        axes[2].hist(bert_probs[mask & (bert_labels==1)], bins=25,
                     alpha=0.5, color=C4, label=f'{lang_name} FAKE', density=True,
                     linestyle='--', histtype='step', linewidth=2)
axes[2].set_title('Score Distribution by Language', color='#dcdcff')
axes[2].set_xlabel('P(FAKE)'); axes[2].legend(framealpha=0.3, fontsize=8)
axes[2].grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('language_analysis.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Language analysis saved → language_analysis.png")



# ## 🔥 14. Attention Visualisation — What mBERT Focuses On
# 
# 
# 

# In[ ]:


def get_attention_weights(model, tokenizer, text, device, max_len=64):
    model.eval()
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    padding='max_length', return_tensors='pt')
    input_ids  = enc['input_ids'].to(device)
    attn_mask  = enc['attention_mask'].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_attentions=True)
    # Last layer, mean over heads, CLS row
    last_attn  = out.attentions[-1][0]          # (heads, seq, seq)
    cls_attn   = last_attn[:, 0, :].mean(0)     # mean over heads → (seq,)
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len    = attn_mask[0].sum().item()
    tokens     = tokens[:seq_len]
    weights    = cls_attn[:seq_len].cpu().numpy()
    weights    = (weights - weights.min())/(weights.max()-weights.min()+1e-8)
    pred_prob  = torch.softmax(out.logits,dim=1)[0,1].item()
    return tokens, weights, pred_prob

demo_articles = [
    ("BREAKING! Scientists DISCOVER that GOVERNMENT is HIDING the CURE for cancer! "
     "Big Pharma DOESN'T want you to KNOW! SHARE before they DELETE this!"),
    ("Researchers at Stanford University published findings in the New England Journal "
     "of Medicine showing that early screening reduces cancer mortality by 35%."),
    ("அதிர்ச்சி! பிரபல நடிகர் இரகசியமாக வெளிநாட்டிற்கு கோடிக்கணக்கான பணம் அனுப்பினார்!"),
    ("தமிழ்நாட்டில் புதிய சுகாதாரக் கொள்கை அமல்படுத்தப்படும் என்று சுகாதார அமைச்சகம் அறிவித்தது."),
]
labels_demo = ["FAKE","REAL","FAKE","REAL"]

fig, axes = plt.subplots(4, 1, figsize=(18, 14))
fig.suptitle('🔥 mBERT Attention Weights — [CLS] Token Focus',
             fontsize=14, color=C4, fontweight='bold')

for ax, text, true_lbl in zip(axes, demo_articles, labels_demo):
    tokens, weights, pred_prob = get_attention_weights(
        bert_model, tokenizer, text, DEVICE)
    pred_lbl = 'FAKE' if pred_prob > 0.5 else 'REAL'
    correct  = pred_lbl == true_lbl

    cmap   = plt.cm.Reds if pred_lbl=='FAKE' else plt.cm.Greens
    colors_attn = [cmap(w) for w in weights]

    x_pos = np.arange(len(tokens))
    ax.bar(x_pos, weights, color=colors_attn, edgecolor='none', width=0.85)
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        ax.text(i, w+0.015, tok[:8], ha='center', va='bottom',
                fontsize=7.5, color='white', rotation=55, fontweight='bold')

    status = '✅' if correct else '❌'
    ax.set_title(
        f'{status} True={true_lbl}  Pred={pred_lbl}  P(FAKE)={pred_prob:.3f}',
        color=C2 if correct else C4, fontsize=10)
    ax.set_xlim(-0.5, len(tokens)-0.5)
    ax.set_ylim(0, 1.5)
    ax.axis('off')

plt.tight_layout()
plt.savefig('attention_viz.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Attention visualisation saved → attention_viz.png")


# ## 🏆 15. Final Model Comparison Dashboard

# In[ ]:


fig = plt.figure(figsize=(18, 11))
fig.suptitle('🏆 Bilingual Fake News Detector — Final Dashboard',
             fontsize=16, color=C3, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ROC all models
ax1 = fig.add_subplot(gs[0,0:2])
ax1.plot([0,1],[0,1],'--',color='#606080',linewidth=1,label='Random')
colors_all = [C3, C2, C1]
for (name, prob, color) in all_models:
    fpr,tpr,_ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax1.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name}  AUC={auc:.4f}')
ax1.fill_between(*roc_curve(y_test, bert_probs)[:2], alpha=0.1, color=C1)
ax1.set_title('ROC Curves — All Models', color='#dcdcff')
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.legend(framealpha=0.3); ax1.grid(True, alpha=0.3)

# Metrics comparison bar
ax2 = fig.add_subplot(gs[0,2])
model_names = ['TF-IDF\n+LR', 'TF-IDF\n+SVM', 'mBERT']
accs = [baseline_results['TF-IDF + LR']['acc'],
        baseline_results['TF-IDF + SVM']['acc'],
        bert_acc]
aucs = [baseline_results['TF-IDF + LR']['auc'],
        baseline_results['TF-IDF + SVM']['auc'],
        bert_auc]
x  = np.arange(len(model_names)); w = 0.38
ax2.bar(x-w/2, accs, w, color=[C3,C2,C1], alpha=0.85, label='Accuracy',
        edgecolor='white', lw=0.4)
ax2.bar(x+w/2, aucs, w, color=[C3,C2,C1], alpha=0.55, label='AUC',
        edgecolor='white', lw=0.4, hatch='//')
ax2.set_xticks(x); ax2.set_xticklabels(model_names, fontsize=9)
ax2.set_ylim(0, 1.15); ax2.legend(framealpha=0.3)
ax2.set_title('Accuracy & AUC', color='#dcdcff'); ax2.grid(True, alpha=0.3, axis='y')
for xi, acc, auc in zip(x, accs, aucs):
    ax2.text(xi-w/2, acc+0.01, f'{acc:.3f}', ha='center', fontsize=8, color='white')
    ax2.text(xi+w/2, auc+0.01, f'{auc:.3f}', ha='center', fontsize=8, color='white')

# Training history
ax3 = fig.add_subplot(gs[1,0])
ep = range(1, len(history['val_auc'])+1)
ax3.plot(ep, history['train_acc'], color=C1, linewidth=2, marker='o', ms=6, label='Train Acc')
ax3.plot(ep, history['val_acc'],   color=C3, linewidth=2, marker='s', ms=6,
         linestyle='--', label='Val Acc')
ax3.plot(ep, history['val_auc'],   color=C4, linewidth=2, marker='^', ms=6,
         linestyle=':', label='Val AUC')
ax3.set_title('mBERT Training History', color='#dcdcff')
ax3.set_xlabel('Epoch'); ax3.set_xticks(list(ep))
ax3.legend(framealpha=0.3); ax3.grid(True, alpha=0.3)

# Score distribution mBERT
ax4 = fig.add_subplot(gs[1,1])
ax4.hist(bert_probs[bert_labels==0], bins=30, alpha=0.65, color=C2,
         label='REAL', density=True)
ax4.hist(bert_probs[bert_labels==1], bins=30, alpha=0.65, color=C4,
         label='FAKE', density=True)
ax4.axvline(0.5, color='white', linewidth=1.5, linestyle='--', label='threshold=0.5')
ax4.set_title('mBERT Score Distribution', color='#dcdcff')
ax4.set_xlabel('P(FAKE)'); ax4.legend(framealpha=0.3); ax4.grid(True, alpha=0.3)

# Summary table
ax5 = fig.add_subplot(gs[1,2])
ax5.axis('off')
summary_data = [
    ['TF-IDF + LR',  f"{baseline_results['TF-IDF + LR']['acc']:.4f}",
     f"{baseline_results['TF-IDF + LR']['f1']:.4f}",
     f"{baseline_results['TF-IDF + LR']['auc']:.4f}"],
    ['TF-IDF + SVM', f"{baseline_results['TF-IDF + SVM']['acc']:.4f}",
     f"{baseline_results['TF-IDF + SVM']['f1']:.4f}",
     f"{baseline_results['TF-IDF + SVM']['auc']:.4f}"],
    ['mBERT',        f'{bert_acc:.4f}', f'{bert_f1:.4f}', f'{bert_auc:.4f}'],
]
tbl = ax5.table(cellText=summary_data,
                colLabels=['Model','Accuracy','F1','AUC'],
                cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.25, 2.1)
for (r,c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1a1a30' if r > 0 else '#2a2a5a')
    cell.set_edgecolor('#4a4a7a'); cell.set_text_props(color='white')
# Highlight mBERT row
for c in range(4):
    tbl[3, c].set_facecolor('#1a3a1a')
ax5.set_title('Model Summary', color='#dcdcff', pad=10)

plt.savefig('final_dashboard.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Final dashboard saved → final_dashboard.png")


# ## 🔎 16. Live Inference Demo — Classify Any Article

# In[ ]:


def predict_article(text, model, tokenizer, device, max_len=128):
    model.eval()
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    padding='max_length', return_tensors='pt')
    with torch.no_grad():
        out   = model(input_ids      = enc['input_ids'].to(device),
                      attention_mask = enc['attention_mask'].to(device))
        probs = torch.softmax(out.logits, dim=1)[0]
    return {'REAL': float(probs[0]), 'FAKE': float(probs[1]),
            'prediction': 'FAKE' if probs[1] > 0.5 else 'REAL',
            'confidence': float(probs.max())}

test_cases = [
    ("SHOCKING: Billionaire SECRETLY funds alien research! Government COVER-UP exposed! "
     "SHARE before DELETED! 90% of scientists SILENCED!", "FAKE"),
    ("The Reserve Bank of India raised interest rates by 25 basis points on Thursday, "
     "citing persistent inflation. The governor confirmed the decision after a three-day review.", "REAL"),
    ("அரசு பள்ளிகளில் இலவச மதிய உணவு திட்டம் விரிவாக்கம் செய்யப்படும் என்று "
     "கல்வி அமைச்சர் திங்கள்கிழமை அறிவித்தார்.", "REAL"),
    ("அதிர்ச்சி! பிரபல நடிகர் இரகசியமாக வெளிநாட்டிற்கு பணம் அனுப்பினார்! "
     "அரசு மறைக்கிறது! உடனே பகிருங்கள்!", "FAKE"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('🔎 mBERT — Live Fake News Detection', fontsize=14,
             color=C1, fontweight='bold')

for ax, (text, true_lbl) in zip(axes.flat, test_cases):
    result = predict_article(text, bert_model, tokenizer, DEVICE)
    pred   = result['prediction']
    conf   = result['confidence']
    correct = pred == true_lbl

    lang = 'Tamil' if any(ord(c) > 0x0B7F for c in text) else 'English'
    bar_colors = [C2 if k=='REAL' else C4 for k in ['REAL','FAKE']]
    vals = [result['REAL'], result['FAKE']]
    bars = ax.barh(['REAL','FAKE'], vals, color=bar_colors,
                   edgecolor='white', linewidth=0.5, alpha=0.85)

    for b, v in zip(bars, vals):
        ax.text(v+0.01, b.get_y()+b.get_height()/2,
                f'{v:.1%}', va='center', fontsize=12,
                color='white', fontweight='bold')

    status  = '✅' if correct else '❌'
    title   = f'{status} [{lang}]  True={true_lbl}  Pred={pred}  Conf={conf:.1%}'
    ax.set_title(title, color=C2 if correct else C4, fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.2)
    snippet = text[:80] + '...' if len(text) > 80 else text
    ax.set_xlabel(f'"{snippet}"', color='#9090b0', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('inference_demo.png', dpi=120, bbox_inches='tight',
            facecolor='#080810', edgecolor='none')
plt.show()
print("✅ Inference demo saved → inference_demo.png")


# ## 🌐 17. Gradio Web App — Deploy on HuggingFace Spaces
# 
# The cell below writes `gradio_app.py`.  
# Run with: `python gradio_app.py`  
# Or push to HuggingFace Spaces for a free public URL.
# 
# ---

# In[ ]:


app_lines = ['import gradio as gr', 'import torch', 'from transformers import AutoTokenizer, AutoModelForSequenceClassification', '', "MODEL_NAME = 'bert-base-multilingual-cased'", 'MAX_LEN    = 128', "DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')", '', "print('Loading model...')", 'tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)', 'model      = AutoModelForSequenceClassification.from_pretrained(', '    MODEL_NAME, num_labels=2)', 'try:', "    model.load_state_dict(torch.load('best_mbert.pt', map_location=DEVICE))", "    print('Loaded fine-tuned weights')", 'except:', "    print('Using base weights (run training notebook first)')", 'model = model.to(DEVICE).eval()', '', 'EXAMPLES = [', "    'BREAKING! Government HIDING cure for cancer! Big Pharma CONSPIRACY! SHARE before deleted!',", "    'WHO published new guidelines on vaccine schedules after review by 50 global experts.',", "    'அதிர்ச்சி! பிரபல நடிகர் இரகசியமாக வெளிநாட்டிற்கு பணம் அனுப்பினார்! உடனே பகிருங்கள்!',", "    'தமிழ்நாட்டில் புதிய சுகாதாரக் கொள்கை அமல்படுத்தப்படும் என்று அமைச்சகம் அறிவித்தது.',", ']', '', 'def classify(text):', '    if not text or not text.strip():', "        return 'Please enter some text.', {}", '    enc = tokenizer(text, max_length=MAX_LEN, truncation=True,', "                    padding='max_length', return_tensors='pt')", '    with torch.no_grad():', "        out   = model(input_ids=enc['input_ids'].to(DEVICE),", "                      attention_mask=enc['attention_mask'].to(DEVICE))", '        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()', '    real_p, fake_p = float(probs[0]), float(probs[1])', "    verdict = 'FAKE NEWS' if fake_p > 0.5 else 'REAL NEWS'", "    emoji   = 'FAKE NEWS' if fake_p > 0.5 else 'REAL NEWS'", "    detail  = f'Confidence: {max(real_p,fake_p):.1%} | Language auto-detected'", "    return f'{emoji} — {verdict}\\n{detail}', {'REAL': real_p, 'FAKE': fake_p}", '', "with gr.Blocks(title='Fake News Detector', theme=gr.themes.Soft()) as demo:", "    gr.Markdown('# Fake News Detector — Tamil & English')", "    gr.Markdown('Powered by **mBERT** fine-tuned on bilingual fake news data')", '    with gr.Row():', '        with gr.Column():', "            txt  = gr.Textbox(label='News Article (Tamil or English)',", "                              placeholder='Paste your news article here...',", '                              lines=6)', "            btn  = gr.Button('Classify', variant='primary')", '            gr.Examples(examples=EXAMPLES, inputs=txt)', '        with gr.Column():', "            out_label = gr.Textbox(label='Verdict')", "            out_conf  = gr.Label(label='Confidence Scores')", '    btn.click(classify, inputs=txt, outputs=[out_label, out_conf])', "    gr.Markdown('*Stage 11 Portfolio Project — mBERT Fine-Tuning*')", '', "if __name__ == '__main__':", '    demo.launch(share=False)']
with open('gradio_app.py', 'w') as _f:
    _f.write('\n'.join(app_lines))
print('✅ Gradio app written → gradio_app.py')
print('Run: python gradio_app.py')
print('Or deploy to HuggingFace Spaces for free public URL')


# 
# ### GitHub README Excerpt
# > Fine-tuned mBERT (bert-base-multilingual-cased) on a synthetic bilingual Tamil + English fake news dataset (4,000 articles). Compared against TF-IDF + LR/SVM baselines. Implemented attention visualisation for explainability and deployed an interactive Gradio app for live classification.
# 
# ### HuggingFace Spaces Deployment
# ```bash
# # 1. Create Space at huggingface.co/spaces
# # 2. Push files
# git init
# git remote add origin https://huggingface.co/spaces/YOUR_NAME/fake-news-detector
# git add gradio_app.py best_mbert.pt requirements.txt
# git commit -m "Deploy mBERT fake news detector"
# git push
# 
# # requirements.txt:
# # transformers gradio torch
# ```
