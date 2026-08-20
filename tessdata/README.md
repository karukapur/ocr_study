# Tesseract language data

This benchmark expects `eng.traineddata` and `chi_tra.traineddata` in this
directory. The binary language models are intentionally ignored by Git. Their
SHA-256 hashes are recorded in every completed run manifest.

The current Traditional Chinese model is the official `tessdata_fast` model:

```bash
curl -L https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_tra.traineddata \
  -o tessdata/chi_tra.traineddata
```

An existing system `eng.traineddata` may be copied or symlinked into this
directory.
