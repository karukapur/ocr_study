# Tesseract language data

This benchmark expects `eng.traineddata` and `chi_tra.traineddata` in this
directory. Their SHA-256 hashes are recorded in every completed run manifest
and verified by the Windows reproducibility workflow.

The current Traditional Chinese model is the official `tessdata_fast` model:

```bash
curl -L https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_tra.traineddata \
  -o tessdata/chi_tra.traineddata
```

An existing system `eng.traineddata` may be copied or symlinked into this
directory for local experiments, but CI needs real checked-in files rather than
machine-local symlinks.
