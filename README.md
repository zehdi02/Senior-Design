# Senior-Design
Github repository of team **MangaVision** for the CCNY Senior Design II (CSC 59867) course.

## Team Members:
| Name | Role | Github |
|------|------|--------|
| Melchizedek De Castro   | Leader   | [zehdi02](https://github.com/zehdi02)   | 
| Johnson Chen   | Systems Savvy   | [JohnsonChen22002](https://github.com/JohnsonChen22002)   |   
| Mahmud Hasan   | Techsmith   | [QuodFinis](https://github.com/QuodFinis)   |  
| Ayon Kumar Das   | Quality Assurance   | [lastMinuteGuy](https://github.com/lastMinuteGuy)  |

## Slides
For a full list of slides/docx, [click here](slides.md).

## Project Summary:
Japanese comic books called Manga are enjoyed by many all around the world. Manga are made of panels with images in them where the reader would read from right to left in a traditional Japanese way. Their wide variety of genres and stories captures the attention and hearts of many readers. However, there are individuals who aren’t able to enjoy these graphic novels because of visual impairments. 

MangaVision was created by passionate manga readers with the goal of spreading their love for manga and making it more accessible, especially for people with visual impairments. We decided to shift our focus from implementing multiple features—such as the Event Drawing Recognizer (EDR), text-to-speech (TTS), and JP-to-EN translation—to concentrating on the EDR. We realized that delivering a strong, well-developed EDR would have the greatest impact and be a crucial milestone for the project.

With MangaVision, we strive to break barriers and foster inclusivity within the manga community and expand outward. 

## Additional Project Information:
Input: A folder consisting of images of each manga page.

Output: An HTML file, which acts as an ereader, w/ buttons such as reading aloud the images and speech bubbles to the reader.

| Dataset | Summary | Element |
|---------|---------|---------|
|Manga109|<li>Provides permission to access 109 manga books legally.</li><li>Provides annotations for character face & body, text boxes, and panel boxes.</li><li>Contains  21,142 pages (2 pages for every image) or 10,602 images</li>|<li>Manga Volumes</li><li>Character faces</li><li>Panel</li><li>Speech/Dialogue</li>|
|KangaiSet|<li>A dataset to supplement Manga109 dataset for facial expression recognition</li><li>Properly annotated according to character’s face bounding boxes from Manga109</li><li>7 emotions: anger, disgust, fear, happiness, neutral, sadness, surprise.</li><li>Annotates 9,387 facial emotions out of the 118,593 faces annotated in Manga109</li>|<li>Character Facial Expression|
|The Quick, Draw! Dataset|<li>A collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!.</li>|<li>Object Classficifation</li>|

| Models/Techniques | Summary | Element |
|---------|---------|---------|
|YOLOv8|<li>Object detection technique used for identification (class recognition) and localization (bounding box) of objects</li><li>Fast and lightweight</li><li>Lower accuracy for smaller objects</li><li>Less precise localization</li><li>May not generalize on complex scene with overlapping objects</li>|<li>Object Detection</li>|
|BLIP| <li>Understand and generate relationships between images and text, excelling in tasks like image captioning. <li>Leverages extensive pretraining on large datasets of image-text pairs</li> <li>Generate detailed, context-aware captions that describe key objects, actions, and relationships in an image|<li>Image Caption/Description</li>|
|DINOv2|<li>Self supervised learning model to extract visual features from images without needing labeled date</li><li>Vision transformers to process images</li><li>Scales with large datasets and often better than CNNs</li><li>Can be used for classification, detection and segmentation</li>|<li>General unlabeled feature extraction for manga panels, characters, and objects</li>|
|DETR (DEtection TRansformer)| <li>Streamlines object detection by removing anchor boxes and non-maximal suppression</li><li>Uses transformer encoder-decoder architecture for object detection tasks</li><li>Eliminates traditional object detection mechanisms (like region proposals or anchor boxes)</li><li>Achieves accuracy and performance comparable to Faster R-CNN</li> | <li>Object Detection</li>|
|Magi| <li>Addresses the challenge of making manga accessible to individuals with visual impairments by generating automatic transcriptions of dialogues and identifying who said what (diarisation)</li><li>Detects panels, text boxes, and character boxes</li><li>Clusters characters by identity and associates dialogues to speakers</li><li>Generates dialogue transcription in the correct reading order</li><li>Annotated evaluation benchmarks using publicly available English manga pages</li> | <li>Text Transcription & Dialogue Association</li> <li>Panel Detection and Ordering</li><li>Text detection and OCR</li>|

# Implementation Goals:
1) Event Drawing Recognizer (Main feature)
- [x] Detect the panel frames on a page
- [x] Detect the text boxes on a page
- [x] Detect character faces/bodies on a page
- [ ] Detect objects in a panel
- [ ] Recognize character facial expressions
- [ ] Recognize character names 
- [ ] Associate dialogues to the correct speaker
- [ ] Establish panels’ correct reading order
- [ ] Caption/Description generation for the panel as a whole
2) Text-To-Speech (Additional Optional Feature)
- [ ] Translate extracted speech bubbles texts to audio                                    
3) JP to EN Translation (Additional Optional Feature)
- [ ] Manga109 dataset is in JP. we need to translate them to EN.

# Challenges:
- Developing zero-shot face recognition model
- Associating dialogues to the correct speaker
- Establishing panels’ correct reading order
- Caption/Description generation for the panel as a whole

## Resources Exploration (Ideas Collection):

## Tools
| Link | Summary |
|------|--------|
| [Optical character recognition for Japanese text for Manga](https://github.com/kha-white/manga-ocr) | <li> Outlined the development of an OCR model designed for manga text recognition, utilizing an approach built on the Transformers' Vision Encoder Decoder framework. </li> <li> OCR model support vertical/horizontal text, furigana annotations, text overlaid on images, various fonts and styles, low-quality images and enables multi-line text recognition by processing entire text bubbles without line splitting. </li> |
| [Text detection](https://github.com/dmMaze/comic-text-detector) | <li> Contained training scripts for a text detector designed for manga which can identify bounding-boxes, text lines, and text segmentation to aid in various translation tasks such as text removal, recognition, and lettering. </li> <li> Current model was trained on approximately 13 thousand anime and comic-style images, utilizing data from Manga109-s, DCM, and synthetic data generated. </li> |
| [Text-To-Speech](https://github.com/mozilla/TTS) | <li> Provides Text-to-Speech (TTS) library which focused on advanced TTS generation, balancing ease of training, speed, and quality. </li> <li> TTS offers pretrained models, tools for dataset quality measurement, and supports over 20 languages for products and research projects </li> <li> Features high-performance Deep Learning models for Text2Speech tasks. </li> |
| [Mokuro - Perform text detection and OCR for each page and generate HTML file](https://github.com/kha-white/mokuro) | <li> Outlined Mokuro, a tool for Japanese learners aiming to read manga in Japanese with the aid of a pop-up dictionary akin to Yomichan. </li> <li> Mokuro conducts text detection and optical character recognition for each page of the manga. </li> <li> Mokuro generates an HTML file that users can open in a browser. </li> |
| [Downloading manga pages as images](https://github.com/manga-download/hakuneko) | <li> Introduced HakuNeko, a cross-platform downloader designed for manga/anime enthusiasts to obtain content from various websites. </li> <li> Assist users in downloading media for situations where offline access is needed. </li> <li> HakuNeko focused on users download content only when they intend to read it, rather than mass downloading thousands of chapters that may never be consumed. </li> |

## Datasets
| Link | Summary | # of Categories | Pros | Cons | Feasibility | 
|------|--------|------|--------|------|--------|
| [Manga109 Dataset](http://www.manga109.org/en/) | Compiled by the Aizawa Yamasaki Matsui Laboratory at the University of Tokyo, consists of 109 manga volumes created by professional Japanese manga artists between the 1970s and 2010s. | 109 | <li> Extensive Collection </li> <li> Permission for Academic/Commercial Use </li> <li> Regular Updates and Corrections </li> | <li> Limited Availability for Commercial Use </li> <li> Potential Data Quality Issues </li> <li> Need Permissions from Authors </li> | <li> Parse and extract relevant information programmatically using XML parsing librariesm or custom scripts. </li> <li> Availability of annotations for each volume of manga in terms of data accessibility </li> <li> Effective integration with Research and Applications </li> 
| [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset) | A collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. | 345 | Generously trained dataset | 345 objects doesn't seem alot, especially for literature |  |

## Techniques
| Paper Title | Summary | Year | Keywords | 
|------|--------|------|--------|
| [Text Extraction and Detection from Images using Machine Learning Techniques: A Research Review](https://ieeexplore.ieee.org/abstract/document/9752274) | To help human beings with a different language from different parts of the world so that they can easily read and understand any language written, and to recognize handwritten text and text captured from images to convert them into digital format. | 2021 | Text extraction, Text detection, Machine Learning Algorithms, Pre-Processing, Segmentation, Optical Character Recognition (OCR) |
| [Object Recognition in Hand Drawn Images Using Machine Ensembling Techniques and Smote Sampling](https://link.springer.com/chapter/10.1007/978-981-15-1384-8_19) | Techniques to classify the stroke-based hand-drawn object. Uses the Quick Draw dataset which is a repository of approximately 50 million hand-drawn drawings of 345 different objects. An approach to the classification of these drawings created using hand strokes. | 2019 | SMOTE, Machine learning, Classiﬁcation, Hyper-parameterselection, Ensembling |
| [Automatic classification of manga characters using density-based clustering](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11515/115150F/Automatic-classification-of-manga-characters-using-density-based-clustering/10.1117/12.2566845.short#_=_) | To classify characters is to get image features from the character's faces and cluster them. To allocate metadata more efficiently, technology that automatically extracts elements such as character and speech is required. | 2020 | Dimension reduction, Communication engineering, Algorithm development, Computer engineering, Detection and tracking algorithms, Facial recognition systems, Feature extraction, Image classification, Computer science, Data conversion |
| [A study on object detection method from manga images using CNN](https://ieeexplore.ieee.org/abstract/document/8369633) | Examines the effectiveness of manga object detection by comparing Fast R-CNN, Faster R-CNN, and SSD. Experimental results show that Fast R-CNN is effective for panel layout and speech balloon, whereas Faster R-CNN is effective for character face and text. | 2018 | Object Detection, Manga, CNN, Fast R-CNN, Faster R-CNN, SSD |
| [DLP-GAN: learning to draw modern Chinese](https://arxiv.org/pdf/2403.03456.pdf) | Unlike previous methods that focus on modern photos to ancient ink paintings, this approach aims to create modern photos from landscape paintings. | 2021 | Unpaired Image Translation, Style Transfer, Asymmetric |


<!-- ARCHIVED
# Features Summary/Design:
A) Manga Panel Drawing/Event Teller - Drawing recognizer:
1. To describe the drawings on each manga panel as an event (must follow manga reading convention of top-right to bottom-left)
2. Such events could be facial expressions, recognizing the subject/character and call their names, actions/verbs being done by a subject, settings or scenes
3. The event would then be transcribed and spoken to the reader through a Text-To-Speech program

B) Simple Text-To-Speech:
1. A button that the reader activates to start reading the manga for them
2. Describing drawings and what’s happening in it in order (top to bottom, right to left)
3. Reads the speech bubbles in order (top to bottom, right to left)

C) Plot Summarizer:
1. A button that allows the reader to summarize the current manga chapter
2. Once the button is pressed, extract all of the text embedded in the speech bubbles using OCR or other text extractor tools
3. Visual events must also be taken into account using our Drawing Recognizer (A)
4. Collect and transcribe every speech and event
5. Put the collection of speech and events in a plot summarizer model
7. The output from the model is then read aloud to the reader with TTS (0)

D) Language Translator:
1. A button that can toggle the current manga chapter into English or to its original language
2. Once the button is pressed, do the same thing as at (B) steps 2-3. (step 3 would already be in english since there would be no original language texts, just images)
3. Collect and transcribe every speech and event in the manga chapter
4. Translate the text from whatever native language it was to English
5. Replace the original text with the translated text while still allowing toggle language swap function

0) Text Extractor
1. With Mokuro, speech bubble detection has already been done for us
2. We just need to extract all the text in the speech bubbles in order (top to bottom, right to left).
3. Perhaps put the extracted text in a separate text file which can be fed onto our Plot Summarizer and Translator models.

Dataset:
<li> [Manga Facial Expressions](https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions) (Original) </li> <li> [Manga Faces Dataset](https://www.kaggle.com/datasets/davidgamalielarcos/manga-faces-dataset) (Extension) </li> | Serves as an expansion of the original Manga Facial Expressions dataset, incorporating additional classes and images which now consists of 670 manga face images categorized into 11 distinct classes. | <li> 462 (Original) </li> <li> 670 (Extension) </li> | <li> Unique Data Source </li> <li> Exhibit a wide range of facial expressions </li> <li> Potential for Algorithmic Development and Testing </li> | <li> Limited Size Dataset </li> <li> Potential Bias or Subjectivity </li> <li> Limited Application Scope </li> | Accurate annotations of the datasets can be used for training machine learning models and conducting reliable analyses.
| [Detecting speech bubbles, japanese text, translating](https://www.kaggle.com/datasets/aasimsani/ampd-base) | The dataset creation process gather Japanese dialogue datasets, sourcing manga-style fonts, identifying speech bubble types, and obtaining manga or black-and-white images for panel filling. | At least 337,000 | <li> Free and publicly available </li> <li> Wide data variety </li> <li> Offers the flexibility for users to customize the dataset creation </li> | <li> Artificially generated which may not fully capture the complexity and diversity of real manga panels </li> <li> Limitations in replicating the nuances of hand-drawn manga art </li> <li> Generating a large-scale manga dataset can be resource-intensive </li> | <li> Translating the manga to English </li> <li> Detect Speech Bubbles </li> <li> Create an artificial manga panel dataset </li> 
| [Image Classifier](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier) | Developing a One Piece character detector following the design of an image classification model. | ≈ 650 | <li> Diversity in appearance, poses, and expressions which can help improve the robustness and generalization of the character detector model. </li> <li> Some images in the dataset have been manually cropped to remove unnecessary noise. </li> <li> Promotes collaboration and knowledge sharing within the community. </li> | <li> Copyright Issues </li> <li> Relatively small data for training a deep learning model </li> <li> Data Quality may vary </li> | <li> Can use this as a reference for character dectection model </li> <li> Can save time and resource by leveraging this dataset </li> <li> Can use it to quickly build and test character detection models as well as to learn some stuff from it </li> |

-->
