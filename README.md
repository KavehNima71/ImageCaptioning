# ImageCaptioning
**Image Captioning** is a field that intersects computer vision and natural language processing (NLP), where the goal is to generate an accurate and meaningful textual description of an image. In other words, the task is to design a model that automatically produces a caption for an image, similar to how a human might describe it.

### **Key Steps and Challenges in Image Captioning:**

**1. Visual Understanding:**
   The model needs to correctly recognize different elements in the image (such as objects, beings, scenes, and their interactions). This is typically done using **Convolutional Neural Networks (CNNs)**, which are well-suited for extracting visual features.
   
**2. Natural Language Generation (NLG):**
   After analyzing the image, the model must convert this information into natural language (e.g., English). This is achieved using **Recurrent Neural Networks (RNNs)** or **Transformer** models, which help generate grammatically and contextually correct sentences.

## **Our Method:**
In this project, the objective is to build an **image captioning system** that automatically generates descriptive captions for images. The system will utilize two major components of deep learning: a **Encoder architecture** for extracting image features and a **Decoder architecture** for generating textual descriptions based on those features.
Encoder-Decoder architecture. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

**Step 1:**

**Encoder architecture:** Feature Extraction using CNN (ResNet50)
For feature extraction, we will leverage the **ResNet50** architecture, a well-known deep CNN model pre-trained on the **ImageNet** dataset. ResNet50 is highly effective in learning hierarchical image representations due to its residual connections, which help in training deeper networks without the problem of vanishing gradients.
The image input will be passed through the **ResNet50** model, and instead of using the final classification layer, we will extract the feature map from one of the intermediate layers (typically the output of the last convolutional block). These feature maps represent a compressed yet informative summary of the image content, which will then be passed to the text generation model.

**Step 2:**

**Decoder architecture:** Text Generation using RNN (LSTM)
Once the image features are extracted, the next step is to generate a natural language description. For this, we will use an **LSTM (Long Short-Term Memory)** network, which is a type of Recurrent Neural Network (RNN) well-suited for sequence prediction tasks. LSTMs are known for their ability to model long-range dependencies in sequences, making them ideal for generating coherent sentences.
The flow for the LSTM network will be as follows:
1. The extracted features from ResNet50 will serve as the initial input to the LSTM network.
2. The LSTM will also receive the previous words generated in the caption (starting with a special '<start>' token) and predict the next word in the sequence.
3. The process continues until the LSTM generates the complete sentence, which ends with a special '<end>' token.


![IMG_20241001_114049_130](https://github.com/user-attachments/assets/3a8564c2-a06d-4331-b961-aa6eecd72e49)


## **Dataset:**
**Flickr8K** is a popular dataset in the field of deep learning, particularly for **image captioning** tasks. It contains a variety of images along with textual descriptions, and it serves as a suitable resource for training deep learning models to generate captions from images.

### **Flickr8K Dataset Specifications:**

**1. Number of Images:**
   - The dataset contains **8,000** images that have been collected from the [**Flickr**](https://hockenmaier.cs.illinois.edu/8k-pictures.html) website. These images are typically of natural and everyday scenes, including humans, animals, and various objects.

**2. Captions:**
   - Each image comes with **five different captions**. These captions are written by humans, and each one provides a unique description of the image's content.
   - This diversity in captions helps models learn to generate creative and varied descriptions and prevents the model from producing stereotypical or limited sentences.

**3. Use Cases:**
   - **Training and evaluating image captioning models:** The dataset is used to train models that are capable of generating captions for images, such as models combining CNNs and RNNs (e.g., LSTM).

**4. Data Format:**
   - The images are provided in **JPEG** format.
   - The captions are provided in a text file, where each line includes the image name and the corresponding caption.

**5. Data Split:**
   - The dataset is typically divided into three main parts:
     - **Training:** About 6,000 images used for training models.
     - **Validation:** Approximately 1,000 images used for initial model evaluation.
     - **Testing:** Around 1,000 images for final evaluation and testing of the model s performance in generating captions.


## **Exploratory Data Analysis:**

Exploratory Data Analysis (EDA) of the Flickr8K dataset is conducted to examine various features of facial images. This dataset contains a variety of images along with textual descriptions, and it serves as a suitable resource for training deep learning models to generate captions from images.

### **Key steps in EDA for this dataset:**


**1. Statistical distribution analysis:** The first step is to examine the number of images available.


![1](https://github.com/user-attachments/assets/a9147f65-76ee-4c40-93fc-d980f5fa6e78)



> Number of images in folder: 8091

> Number of images in token txt: 8092.0

> Total number of captions: 40460

> Captions per image: 5.0




**2. Anomaly detection:** Identify mislabeled images or duplicate data and clean the dataset.

> image name not in folder: 2258277193_586949ec62.jpg.1



**3. Analyzing Caption Length:** 

   **- Number of Words per Caption:** One key analysis in EDA is to examine the average length of captions. This will give you an idea of the distribution of caption lengths. For example, captions may range from 5 to 20 words.
   

> Total Words: 476878

> Average words per Caption: 11.786406327236778

> Maximum caption length: 38

> Minimum caption length: 1


   **- Length Distribution:** Plotting a histogram of caption word counts can show whether the captions are generally short or long. This is important when designing language models like LSTMs, as the model needs to handle varying sentence lengths.


![2](https://github.com/user-attachments/assets/74342624-2958-4268-ab98-66a2bdbc29a9)


**4. Image Quality and Size:** Image quality and size are critical factors. The images may come in various dimensions, so it's important to check if the image sizes are consistent.  If images have varying dimensions, you might need **preprocessing** to resize them to standard input dimensions for CNNs, typically 224x224 pixels.


![3](https://github.com/user-attachments/assets/e7ca6410-ec61-4b85-9f8a-3ea9e2bc1d45)


Performing **Exploratory Data Analysis (EDA)** on the **Flickr8K** dataset allows you to gain a better understanding of the structure, quality, and distribution of the data.
