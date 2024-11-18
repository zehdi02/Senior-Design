## MangaVision Pipeline:
1. One single-page image is passed onto our MangaVision model.
2. Prediction results are passed onto the panel and text box sorting pipeline ("The Manga Whisperer" research paper).
3. The sorted panels and text boxes are then passed on to an OCR (manga_ocr).
4. Sorted texts are transcribed into a text file.
