# Senior-Design
Github repository of team **MangaVision** for the CCNY Senior Design 59866 course.

## Team Members:
| Name | Role | Github |
|------|------|--------|
| Melchizedek De Castro   | Leader   | [zehdi02](https://github.com/zehdi02)   | 
| Johnson Chen   | Systems Savvy   | [JohnsonChen22002](https://github.com/JohnsonChen22002)   |   
| Mahmud Hasan   | Techsmith   | [QuodFinis](https://github.com/QuodFinis)   |  
| Ayon Kumar Das   | Quality Assurance   | [lastMinuteGuy](https://github.com/lastMinuteGuy)  |

Team Introduction Slide: [Link](https://docs.google.com/presentation/d/1WNn4oexdCydlKBAyXx5dTc79Uo_LxP4TgfdswozQwrA/edit?usp=sharing)

Outline and Ideas: [Link](https://docs.google.com/document/d/1Q3Uw8UuIPxLry2x__Ho96tgG0YmFKRJOzUxRCji3SqQ/edit)

## Project Description:
MangaVision is made by passionate manga readers whose aim is to spread their love for manga and make it accessible for people with visual disabilties or impairments.

MangaVision's main feature is to audibly describe the drawn visual events depicted on the manga pages, empowering the user to enjoy manga just like every other manga lovers out there. MangaVision also enables users to have image speech bubbles read aloud and offer the convenience of summarizing the overall plot of chapters/volumes enhancing understanding and engagement, as well as language translation.

Potential Key features of MangaVision include leveraging OCR technology to seamlessly scan and edit texts within image speech bubbles. This capability opens up new possibilities for accessibility and customization. Furthermore, our platform facilitates the translation of native manga texts from Japanese to English, broadening the reach and appeal of manga content.

With MangaVision, we strive to break barriers and foster inclusivity within the manga community and expanding outward. 

## Key Features:
0) Simple Text-To-Speech:
1. A button that the reader activates to start reading the manga for them
2. Describing drawings and what’s happening in it in order (top to bottom, right to left)
3. Reads the speech bubbles in order (top to bottom, right to left)

A) Manga Panel Drawing/Event Teller - Drawing recognizer:
1. To describe the drawings on each manga panel as an event (must follow manga reading convention of top-right to bottom-left)
2. Such events could be facial expressions, recognizing the subject/character and call their names, actions/verbs being done by a subject, settings or scenes
3. The event would then be transcribed and spoken to the reader through a Text-To-Speech program

B) Plot Summarizer:
1. A button that allows the reader to summarize the current manga chapter
2. Once the button is pressed, extract all of the text embedded in the speech bubbles using OCR or other text extractor tools
3. Visual events must also be taken into account using our Drawing Recognizer (A)
4. Collect and transcribe every speech and event
5. Put the collection of speech and events in a plot summarizer model
7. The output from the model is then read aloud to the reader with TTS (0)

C) Language Translator:
1. A button that can toggle the current manga chapter into English or to its original language
2. Once the button is pressed, do the same thing as at (B) steps 2-3. (step 3 would already be in english since there would be no original language texts, just images)
3. Collect and transcribe every speech and event in the manga chapter
4. Translate the text from whatever native language it was to English
5. Replace the original text with the translated text while still allowing toggle language swap function

D) Text Extractor
1. With Mokuro, speech bubble detection has already been done for us
2. We just need to extract all the text in the speech bubbles in order (top to bottom, right to left).
3. Perhaps put the extracted text in a separate text file which can be fed onto our Plot Summarizer and Translator models.



## Resources Exploration (Ideas Collection):

## Tools
| Link | Summary |
|------|--------|
| <li> [Optical character recognition for Japanese text for Manga](https://github.com/kha-white/manga-ocr) </li> | <li> Outlined the development of an OCR model designed for manga text recognition, utilizing an approach built on the Transformers' Vision Encoder Decoder framework. </li> <li> OCR model support vertical/horizontal text, furigana annotations, text overlaid on images, various fonts and styles, low-quality images and enables multi-line text recognition by processing entire text bubbles without line splitting. </li> |
| <li> [Text detection](https://github.com/dmMaze/comic-text-detector) </li> | <li> Contained training scripts for a text detector designed for manga which can identify bounding-boxes, text lines, and text segmentation to aid in various translation tasks such as text removal, recognition, and lettering. </li> <li> Current model was trained on approximately 13 thousand anime and comic-style images, utilizing data from Manga109-s, DCM, and synthetic data generated. </li> |
| <li> [Text-To-Speech](https://github.com/mozilla/TTS) </li> | <li> Provides Text-to-Speech (TTS) library which focused on advanced TTS generation, balancing ease of training, speed, and quality. </li> <li> TTS offers pretrained models, tools for dataset quality measurement, and supports over 20 languages for products and research projects </li> <li> Features high-performance Deep Learning models for Text2Speech tasks. </li> |
| <li> [Mokuro - Perform text detection and OCR for each page and generate HTML file](https://github.com/kha-white/mokuro) </li> | <li> Outlined Mokuro, a tool for Japanese learners aiming to read manga in Japanese with the aid of a pop-up dictionary akin to Yomichan. </li> <li> Mokuro conducts text detection and optical character recognition for each page of the manga. </li> <li> Mokuro generates an HTML file that users can open in a browser. </li> |
| <li> [Downloading manga pages as images](https://github.com/manga-download/hakuneko) </li> | <li> Introduced HakuNeko, a cross-platform downloader designed for manga/anime enthusiasts to obtain content from various websites. </li> <li> Assist users in downloading media for situations where offline access is needed. </li> <li> HakuNeko focused on users download content only when they intend to read it, rather than mass downloading thousands of chapters that may never be consumed. </li> |

## Datasets
| Link | Summary | # of Categories | Pros | Cons | Feasibility | 
|------|--------|------|--------|------|--------|
| [Manga109 Dataset](http://www.manga109.org/en/) | <li> Compiled by the Aizawa Yamasaki Matsui Laboratory at the University of Tokyo, consists of 109 manga volumes created by professional Japanese manga artists between the 1970s and 2010s. </li> | <li> 109 </li> | <li> Extensive Collection </li> <li> Permission for Academic/Commercial Use </li> <li> Regular Updates and Corrections </li> | <li> Limited Availability for Commercial Use </li> <li> Potential Data Quality Issues </li> <li> Need Permissions from Authors </li> | <li> Parse and extract relevant information programmatically using XML parsing librariesm or custom scripts. </li> <li> Availability of annotations for each volume of manga in terms of data accessibility </li> <li> Effective integration with Research and Applications </li> 
| <li> [Manga Facial Expressions](https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions) (Original) </li> <li> [Manga Faces Dataset](https://www.kaggle.com/datasets/davidgamalielarcos/manga-faces-dataset) (Extension) </li> | <li> Second link serves as an expansion of the original Manga Facial Expressions dataset, incorporating additional classes and images which now consists of 670 manga face images categorized into 11 distinct classes. </li> | <li> 462 (Original) </li> <li> 670 (Extension) </li> | <li> Unique Data Source </li> <li> Exhibit a wide range of facial expressions </li> <li> Potential for Algorithmic Development and Testing </li> | <li> Limited Size Dataset </li> <li> Potential Bias or Subjectivity </li> <li> Limited Application Scope </li> | <li> Accurate annotations of the datasets can be used for training machine learning models and conducting reliable analyses. </li>
| <li> [Detecting speech bubbles, japanese text, translating](https://www.kaggle.com/datasets/aasimsani/ampd-base) </li> | <li> The dataset creation process gather Japanese dialogue datasets, sourcing manga-style fonts, identifying speech bubble types, and obtaining manga or black-and-white images for panel filling. </li> | <li> At least 337,000 </li> | <li> Free and publicly available </li> <li> Wide data variety </li> <li> Offers the flexibility for users to customize the dataset creation </li> | <li> Artificially generated which may not fully capture the complexity and diversity of real manga panels </li> <li> Limitations in replicating the nuances of hand-drawn manga art </li> <li> Generating a large-scale manga dataset can be resource-intensive </li> | <li> Translating the manga to English </li> <li> Detect Speech Bubbles </li> <li> Create an artificial manga panel dataset </li> 
| <li> [Image Classifier](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier) </li> | <li> Developing a One Piece character detector following the design of an image classification model. </li>| <li> ≈ 650 </li> | <li> Diversity in appearance, poses, and expressions which can help improve the robustness and generalization of the character detector model. </li> <li> Some images in the dataset have been manually cropped to remove unnecessary noise. </li> <li> Promotes collaboration and knowledge sharing within the community. </li> | <li> Copyright Issues </li> <li> Relatively small data for training a deep learning model </li> <li> Data Quality may vary </li> | <li> Can use this as a reference for character dectection model </li> <li> Can save time and resource by leveraging this dataset </li> <li> Can use it to quickly build and test character detection models as well as to learn some stuff from it </li> |

<!-- ARCHIVED --!>
<!-- ### Documentations
[Object Detection for Comics using Manga109 Annotations:](https://arxiv.org/pdf/1803.08670.pdf)
<p>The article introduces solutions for object detection in comics, notably the Manga109-annotations dataset and the SSD300-fork method. Created over eight months, Manga109-annotations provides comprehensive annotations for bounding boxes, character names, and text contents. SSD300-fork addresses assignment issues by replicating the detection layer for each category, outperforming other CNN-based methods with a 3% mAP improvement and a 9% boost in face detection accuracy over SSD300. Application of SSD300-fork to eBDtheque demonstrates significant advancements in body detection compared to existing methods. </p>

[Sketch-based manga retrieval using manga109 dataset](https://link.springer.com/content/pdf/10.1007/s11042-016-4020-z.pdf)
<p> The article presents a comprehensive sketch-based manga retrieval system along with novel query methodologies, featuring margin area labeling, EOH feature description with screen tone removal, and approximate nearest-neighbor search using product quantization. It introduces the Manga109 dataset, comprising 21,142 manga images drawn by 94 professional artists, making it the largest manga image dataset available for research. Experimental results demonstrate the system's efficiency and scalability, achieving rapid retrieval from a vast number of pages. Notably, the system captures author characteristics through edge histogram features, enabling retrieval of characters drawn by the same artist. Furthermore, query interactions like relevance feedback facilitate content-based searches, retrieving specific character expressions across various manga titles. The paper suggests future directions involving the integration of sketch and keyword-based searches, promising further advancements in manga retrieval technology. </p>

[Building a Manga Dataset ”Manga109” with Annotations for Multimedia Applications](https://arxiv.org/pdf/2005.04425.pdf)
<p> The article introduce Manga109, consisting of 109 Japanese comic books with annotations for frames, speech texts, character faces, and bodies, totaling over 500k annotations, facilitating machine learning algorithms and evaluation. Additionally, a subset is available for industrial use. Text detection using a Single Shot Multibox Detector (SSD) achieved high accuracy, with an AP of 0.918 for SSD512. Sketch-based manga retrieval compared edge orientation histograms (EOHs) and deep features, with deep features outperforming significantly. Character face generation using Progressive Growing of GANs (PGGAN) produced high-quality results, demonstrating the utility of Manga109 for various multimedia applications. </p>

[Manga109 Dataset and Creation of Metadata](https://dl.acm.org/doi/pdf/10.1145/3011549.3011551)
<p> The article discusses the creation of the Manga109 dataset, which comprises 109 Japanese comic books available for academic use, addressing the need for publicly available datasets with detailed annotations for comic image processing. The authors present an ongoing project aimed at constructing metadata for Manga109, defining metadata elements such as frames, texts, and characters, along with guidelines to enhance annotation quality. They introduce a web-based annotation tool designed for efficient metadata creation and evaluate its effectiveness through user studies. The dataset covers a wide range of genres and publication years, spanning from the 1970s to the 2010s, with permissions obtained from creators for research purposes. The paper emphasizes the importance of such datasets for machine learning algorithms and method evaluations in comic image processing, providing valuable insights into the annotation process and software design. </p>

[Manga109Dialog: A Large-scale Dialogue Dataset for Comics Speaker Detection](https://arxiv.org/pdf/2306.17469.pdf)
<p> The article introduces Manga109Dialog, the largest dialogue dataset for comics speaker detection, addressing the growing need for automated methods to analyze e-comics. Recognizing the limitations of existing annotations, the dataset is meticulously constructed, linking text to character bounding boxes and categorizing annotations based on prediction difficulty. The proposed approach leverages deep learning and scene graph generation models, enhanced by considering frame information to capture the unique structure of comics. Experimental results demonstrate significant improvements over rule-based methods, with qualitative examples showcasing the effectiveness of the proposed approach. Challenges and future directions, including the potential incorporation of natural language processing, are highlighted, emphasizing the dataset's reliability and the method's superiority in comics speaker detection, laying the groundwork for future research in this field. </p>

[A Method to Annotate Who Speaks a Text Line in Manga and Speaker-Line Dataset for Manga109](https://dl.nkmr-lab.org/papers/403/paper.pdf)
<p> The article outlines a method for annotating speakers in manga text lines and presents a corresponding dataset for Manga109. It introduces challenges in accurately recognizing speakers and highlights the importance of annotated datasets for research. The proposed method involves dragging text lines onto character faces to assign speakers, with a prototype system developed for implementation. The dataset, constructed with contributions from 56 annotators, facilitates speaker-line mapping. Analysis reveals a decreasing perfect match rate with increasing annotators and introduces Evaluation Consistency Indicators (ECI) to assess speaker mapping quality. Results show variation in difficulty across comics, particularly in scenes like battles and dark settings. The document suggests strategies for annotator allocation based on scene complexity and proposes future directions for automatic speaker judgment and dynamic annotation requirements. </p>

[The Manga Whisperer: Automatically Generating Transcriptions for Comics](https://arxiv.org/pdf/2401.10224.pdf)
<p> The article presents an algorithm for automatically transcribing manga comics into text to improve accessibility for visually impaired readers. It outlines a method to construct a directed acyclic graph (DAG) to determine the reading order of panels based on manga layout conventions, considering factors like panel positions and overlaps. Supplementary materials include detailed descriptions of the algorithm for ordering panels, the PopManga dataset and its annotation process, character clustering evaluation methods, and the OCR model trained using synthetic data. These materials provide comprehensive insights into the methodology, dataset creation, annotation procedures, and model training involved in making manga more accessible to a wider audience. </p>

[Complex Character Retrieval from Comics using Deep Learning](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2019_IEICE_GC_bs_04_018.pdf)
<p> The article explores the application of deep learning techniques, particularly the You Only Look Once (YOLO) algorithm, for object detection within digital comic books. It addresses the challenge of character-based search in these comics, which differ significantly from real-life objects, presenting complex visual structures that make detection more challenging. Through experiments conducted on the Manga109 dataset, comprising over 10,000 annotated images, the study demonstrates high accuracy in detecting text, frames, faces, and bodies using YOLOv3, achieving notable average precision values. However, when tested on the eBDtheque dataset, which features more diverse and complex characters, detection accuracy slightly decreases. The paper concludes by highlighting the need for larger datasets encompassing various character types to develop a more robust information retrieval system for comics, envisioning the potential for advanced search functionalities based on character-related input, which could significantly enhance user experience in navigating digital comic books. </p> -->
