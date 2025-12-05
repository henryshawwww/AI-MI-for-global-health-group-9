# AI-MI-for-global-health-group-9
# AI-Assisted VIA for Cervical Cancer Screening

End-to-end PyTorch pipeline to train and evaluate a 3-class classifier for cervix images
(negative / positive / suspiciousCancer) using EfficientNet-B3, with data stored on
Google Drive and code in a Jupyter / Colab notebook.

---

## 1. Quick Start (Google Colab – Recommended)

1. **Open the code repository** (this project) in your browser.
2. Make sure the notebook file `aimi_projectvia.ipynb` is in the repo root.
3. **Open the notebook in Colab**  
   - Click on the notebook in GitHub → `Open in Colab`  
   - or upload `aimi_projectvia.ipynb` directly to your Google Drive and open with Colab.
4. **Enable GPU** in Colab  
   `Runtime → Change runtime type → Hardware accelerator → GPU → Save`.
5. **Download and place the dataset** (see Section 3) into your Google Drive so that it lives at:
6. In Colab, run all cells in order (`Runtime → Run all`).  
At the end you should see:
- Printed dataset statistics
- Training / validation curves
- A saved checkpoint `best_model_balanced.pth` in `MyDrive`
- Example predictions on 3–5 test images with confidence scores.

For more detail and troubleshooting, follow the sections below.

---

## 2. Repository Contents

Expected contents of the public code repository:

- `README.md` – this file, with step-by-step instructions.
- `aimi_projectvia.ipynb` – main notebook:
- Data loading & augmentation  
- EfficientNet-B3 model definition  
- Training with class-weighted loss, mixup, scheduler & early stopping  
- Evaluation & visualization of sample predictions
- `expected_outputs.pdf` – PDF of expected results/outputs (see Section 6).
- (Optional but recommended) `sample_data/` – a very small subset of non-PHI images
for a quick smoke test if the full dataset is not downloaded.

All core usage and sample test code is contained in `aimi_projectvia.ipynb` in an
.ipynb format, as required.

---

## 3. Dataset Setup

### 3.1 Download the dataset

The full cervix dataset used in this project is shared via Google Drive:

**Public data link (anyone can view & download):**  
`https://drive.google.com/drive/folders/16AfqE7oHk2e6MWFt85pOip-h9VAA6jpx?usp=sharing`

1. Open the link in a browser.
2. Download the dataset folder (or the zip file, depending on how it is stored).
3. After unzipping (if necessary), ensure you have a folder named **`dataset_split`**.
4. Upload the `dataset_split` folder to the root of your Google Drive, so in Colab
it is visible as:


### 3.2 Expected directory structure

Inside `dataset_split`, the notebook assumes the following structure:

```text
dataset_split/
train/
 negative/
   *.jpg / *.jpeg / *.png
 positive/
   *.jpg / *.jpeg / *.png
 suspiciousCancer/
   *.jpg / *.jpeg / *.png

val/
 negative/
 positive/
 suspiciousCancer/

test/
 negative/
 positive/
 suspiciousCancer/

✓ dataset_split folder found
  ✓ train folder found
    - negative: 104 images
    - positive: 89 images
    - suspiciousCancer: 18 images
  ✓ val folder found
    - negative: 22 images
    - positive: 19 images
    - suspiciousCancer: 4 images
  ✓ test folder found
    - negative: 22 images
    - positive: 19 images
    - suspiciousCancer: 4 images
Class mapping: {'negative': 0, 'positive': 1, 'suspiciousCancer': 2}
Train: 211 images, Val: 45 images, Test: 45 images
Class distribution: Counter({0: 104, 1: 89, 2: 18})
Class weights: tensor([0.6763, 0.7903, 3.9074])
