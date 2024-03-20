# Senior-Design
Github repository of team **MangaVision** for the CCNY Senior Design 59866 course.
## Members:
[Melchizedek De Castro (Leader)](https://github.com/zehdi02)\
[Johnson Chen (Systems Savvy)](https://github.com/JohnsonChen22002)\
[Mahmud Hasan (Techsmith)](https://github.com/QuodFinis)\
[Ayon Kumar Das (Quality Assurance)](https://github.com/lastMinuteGuy)

## Project Description

Introducing MangaVision, a impactful project with a lot of potentials, designed with manga enthusiasts. Our passion for reading manga drives us to create a solution that enhances accessibility and enjoyment for all especially individuals with visual impairments in mind.

MangaVision aims to provide a comprehensive experience by describing visual events within manga pages to the reader. Through innovative technology, we enable users to have image speech bubbles read aloud and offer the convenience of summarizing the overall plot of chapters/volumes, enhancing understanding and engagement.

Potential Key features of MangaVision include leveraging OCR technology to seamlessly scan and edit texts within image speech bubbles. This capability opens up new possibilities for accessibility and customization. Furthermore, our platform facilitates the translation of native manga texts from Japanese to English, broadening the reach and appeal of manga content.

With MangaVision, we strive to break barriers and foster inclusivity within the manga community and expanding outward. 

## Team Introduction Slides (Assignment 0)
link: https://docs.google.com/presentation/d/1WNn4oexdCydlKBAyXx5dTc79Uo_LxP4TgfdswozQwrA/edit?usp=sharing

## Resources Exploration (Ideas Collection)

### Documentations
[Object Detection for Comics using Manga109 Annotations:](https://arxiv.org/pdf/1803.08670.pdf)
<details>
  <summary>Summary (click to expand)</summary>
  <p>The article introduces solutions for object detection in comics, notably the Manga109-annotations dataset and the SSD300-fork method. Created over eight months, Manga109-annotations provides comprehensive annotations for bounding boxes, character names, and text contents. SSD300-fork addresses assignment issues by replicating the detection layer for each category, outperforming other CNN-based methods with a 3% mAP improvement and a 9% boost in face detection accuracy over SSD300. Application of SSD300-fork to eBDtheque demonstrates significant advancements in body detection compared to existing methods. </p>
</details>

[Sketch-based manga retrieval using manga109 dataset](https://link.springer.com/content/pdf/10.1007/s11042-016-4020-z.pdf)
<details>
  <summary>Summary (click to expand)</summary>
  <p> The article presents a comprehensive sketch-based manga retrieval system along with novel query methodologies, featuring margin area labeling, EOH feature description with screen tone removal, and approximate nearest-neighbor search using product quantization. It introduces the Manga109 dataset, comprising 21,142 manga images drawn by 94 professional artists, making it the largest manga image dataset available for research. Experimental results demonstrate the system's efficiency and scalability, achieving rapid retrieval from a vast number of pages. Notably, the system captures author characteristics through edge histogram features, enabling retrieval of characters drawn by the same artist. Furthermore, query interactions like relevance feedback facilitate content-based searches, retrieving specific character expressions across various manga titles. The paper suggests future directions involving the integration of sketch and keyword-based searches, promising further advancements in manga retrieval technology. </p>
</details>

[Building a Manga Dataset ”Manga109” with Annotations for Multimedia Applications](https://arxiv.org/pdf/2005.04425.pdf)
<details>
  <summary>Summary (click to expand)</summary>
  <p> The article introduce Manga109, consisting of 109 Japanese comic books with annotations for frames, speech texts, character faces, and bodies, totaling over 500k annotations, facilitating machine learning algorithms and evaluation. Additionally, a subset is available for industrial use. Text detection using a Single Shot Multibox Detector (SSD) achieved high accuracy, with an AP of 0.918 for SSD512. Sketch-based manga retrieval compared edge orientation histograms (EOHs) and deep features, with deep features outperforming significantly. Character face generation using Progressive Growing of GANs (PGGAN) produced high-quality results, demonstrating the utility of Manga109 for various multimedia applications. </p>
</details>

[Manga109 Dataset and Creation of Metadata](https://dl.acm.org/doi/pdf/10.1145/3011549.3011551)
<details>
  <summary>Summary (click to expand)</summary>
  <p> The article discusses the creation of the Manga109 dataset, which comprises 109 Japanese comic books available for academic use, addressing the need for publicly available datasets with detailed annotations for comic image processing. The authors present an ongoing project aimed at constructing metadata for Manga109, defining metadata elements such as frames, texts, and characters, along with guidelines to enhance annotation quality. They introduce a web-based annotation tool designed for efficient metadata creation and evaluate its effectiveness through user studies. The dataset covers a wide range of genres and publication years, spanning from the 1970s to the 2010s, with permissions obtained from creators for research purposes. The paper emphasizes the importance of such datasets for machine learning algorithms and method evaluations in comic image processing, providing valuable insights into the annotation process and software design. </p>
</details>

[Manga109Dialog: A Large-scale Dialogue Dataset for Comics Speaker Detection](https://arxiv.org/pdf/2306.17469.pdf)
<details>
  <summary>Summary (click to expand)</summary>
  <p> The paper introduces Manga109Dialog, the largest dialogue dataset for comics speaker detection, addressing the growing need for automated methods to analyze e-comics. Recognizing the limitations of existing annotations, the dataset is meticulously constructed, linking text to character bounding boxes and categorizing annotations based on prediction difficulty. The proposed approach leverages deep learning and scene graph generation models, enhanced by considering frame information to capture the unique structure of comics. Experimental results demonstrate significant improvements over rule-based methods, with qualitative examples showcasing the effectiveness of the proposed approach. Challenges and future directions, including the potential incorporation of natural language processing, are highlighted, emphasizing the dataset's reliability and the method's superiority in comics speaker detection, laying the groundwork for future research in this field. </p>
</details>

[A Method to Annotate Who Speaks a Text Line in Manga and Speaker-Line Dataset for Manga109](https://dl.nkmr-lab.org/papers/403/paper.pdf)

[The Manga Whisperer: Automatically Generating Transcriptions for Comics](https://arxiv.org/pdf/2401.10224.pdf)

[Complex Character Retrieval from Comics using Deep Learning](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2019_IEICE_GC_bs_04_018.pdf)

### Tools
[Optical character recognition for Japanese text for Manga](https://github.com/kha-white/manga-ocr)

[Text detection](https://github.com/dmMaze/comic-text-detector)

[Text-To-Speech](https://github.com/mozilla/TTS)

[Mokuro - Perform text detection and OCR for each page and generate HTML file](https://github.com/kha-white/mokuro)

[Downloading manga pages as images](https://github.com/manga-download/hakuneko)

### Datasets
[Manga Facial Expressions](https://www.kaggle.com/datasets/mertkkl/manga-facial-expressions)

[Manga Faces Dataset](https://www.kaggle.com/datasets/davidgamalielarcos/manga-faces-dataset)

[Detecting speech bubbles, japanese text, translating](https://www.kaggle.com/datasets/aasimsani/ampd-base)

[Image Classifier](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier)

### Outline and Ideas

[SD Outline](https://docs.google.com/document/d/1Q3Uw8UuIPxLry2x__Ho96tgG0YmFKRJOzUxRCji3SqQ/edit)
