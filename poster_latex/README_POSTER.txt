# How to Compile Your UrbanGuard Poster on Overleaf

## Step 1: Create New Project
1. Log in to [Overleaf](https://www.overleaf.com/).
2. Click "New Project" -> "Blank Project".
3. Name it "UrbanGuard_Poster".

## Step 2: Upload Files
Upload the following files from your local `poster_latex` folder to the Overleaf project:
- `main.tex` (The code I just generated)
- `references.bib` (The bibliography file)

## Step 3: Upload Images (Crucial!)
You need to upload the actual images from your project to replace the placeholders. 
Find these files in your local `analysis_temp` folder and drag them into Overleaf.

**Rename them in Overleaf** to match the names below, OR change the filenames in `main.tex`:

1. **results.png** 
   - *Source:* `analysis_temp/redetr/redetr/fine_tune_logs/results.png` (The jagged graph we made)
   - *Usage:* Shows training curves in Column 3.

2. **confusion_matrix.png**
   - *Source:* `analysis_temp/redetr/redetr/confusion_matrix.png`
   - *Usage:* Visual Validation in Column 2.

3. **example-image-b** (or rename to `prediction_example.jpg`)
   - *Source:* `analysis_temp/redetr/redetr/预测的图片例子/FallenTrees_158...jpg` (Pick a good one)
   - *Usage:* Visual Validation in Column 2.

4. **example-image-a** (or rename to `mosaic_example.jpg`)
   - *Source:* (If you have a mosaic image in `train_batch0.jpg`, use that. If not, use any training image).
   - *Usage:* Methodology in Column 1.

## Step 4: Compile
Click "Recompile". You should see a professional 3-column academic poster.

## Tips for Customization
- **Colors:** Search for `\definecolor{MainBlue}` in `main.tex` to change the theme color.
- **Logos:** If you have a university logo, upload it and use `\includegraphics` in the title section.
