# Master the basics of Azure: AI Fundamentals (AI-900)

## Skills at a glance
 - Describe Artificial Intelligence workloads and considerations (15–20%)

 - Describe fundamental principles of machine learning on Azure (20–25%)

 - Describe features of computer vision workloads on Azure (15–20%)

 - Describe features of Natural Language Processing (NLP) workloads on Azure (15–20%)

 - Describe features of generative AI workloads on Azure (15–20%)

## Describe Artificial Intelligence workloads and considerations (15–20%)
### Identify features of common AI workloads
 - Identify features of data monitoring and anomaly detection workloads
    - data monitoring: track and analyze the data used to train and test AI models. This includes monitoring the quality, accuracy, and consistency of the data, as well as identifying and addressing any biases or errors in the data that could impact the performance of the AI system.
    - anomaly detection: examine specific data points and detect rare occurrences that seem suspicious because they’re different from the established pattern of behaviors.
   
 - Identify features of content moderation and personalization workloads
   - content moderation: help moderate large and complex volumes of user-generated content (UGC) and reclaim up to 95% of the time of moderating content manually.
   - personalization: deliver higher-quality personalized experiences for customers across digital channels, all tailored to business needs.

 - Identify computer vision workloads
   - Image classification: Image classification involves training a machine learning model to classify images based on their contents.
   - Object detection: Object detection machine learning models are trained to classify individual objects within an image, and identify their location with a bounding box.
   - Semantic segmentation: Semantic segmentation is an advanced machine learning technique in which individual pixels in the image are classified according to the object to which they belong.
   - Image analysis: You can create solutions that combine machine learning models with advanced image analysis techniques to extract information from images, including "tags" that could help catalog the image or even descriptive captions that summarize the scene shown in the image.
   - Face detection, analysis, and recognition: Face detection is a specialized form of object detection that locates human faces in an image. This can be combined with classification and facial geometry analysis techniques to recognize individuals based on their facial features.
   - Optical character recognition (OCR): Optical character recognition is a technique used to detect and read text in images. You can use OCR to read text in photographs (for example, road signs or store fronts) or to extract information from scanned documents such as letters, invoices, or forms.
   - **Azure AI Vision features:**
     - Image Analysis
     - Face
     - Optical Character Recognition (OCR)

 - Identify natural language processing workloads
   - Analyze and interpret text in documents, email messages, and other sources.
   - Interpret spoken language, and synthesize speech responses.
   - Automatically translate spoken or written phrases between languages.
   - Interpret commands and determine appropriate actions.
   - **Azure AI Language:** understanding and analyzing text, training conversational language models that can understand spoken or text-based commands, and building intelligent applications.
   - **Azure AI Speech:** speech recognition and synthesis, real-time translations, conversation transcriptions, and more.

 - Identify knowledge mining workloads
   - Extract information from large volumes of often unstructured data to create a searchable knowledge store.
   - **Azure AI Search:** a private, enterprise, search solution that has tools for building indexes. The indexes can then be used for internal only use, or to enable searchable content on public facing internet assets.

 - Identify document intelligence workloads
   - Document Intelligence is the area of AI that deals with managing, processing, and using high volumes of a variety of data found in forms and documents. Document intelligence enables you to create software that can automate processing for contracts, health documents, financial forms and more.
   - **Azure AI Document Intelligence:** manage and accelerate data collection from scanned documents.

 - Identify features of generative AI workloads
   - Generative AI applications take in natural language input, and return appropriate responses in a variety of formats including natural language, image, code, and audio.
   - **Azure OpenAI service:** Microsoft's cloud solution for deploying, customizing, and hosting generative AI models. It brings together the best of OpenAI's cutting edge models and APIs with the security and scalability of the Azure cloud platform.

### Identify guiding principles for responsible AI
 - Describe considerations for fairness in an AI solution
   - AI systems should treat all people fairly without bias( gender, ethnicity, or other factors that result in an unfair advantage or disadvantage to specific groups of applicants).
   - Azure Machine Learning includes the capability to interpret models and quantify the extent to which each feature of the data influences the model's prediction. This capability helps data scientists and developers identify and mitigate bias in the model.

 - Describe considerations for reliability and safety in an AI solution
   - AI systems should perform reliably and safely. Unreliability in systems can result in substantial risk to human life. AI-based software application development must be subjected to rigorous testing and deployment management processes to ensure that they work as expected before release.

 - Describe considerations for privacy and security in an AI solution
   - AI systems should be secure and respect privacy.  The machine learning models on which AI systems are based rely on large volumes of data, which may contain personal details that must be kept private. Even after the models are trained and the system is in production, privacy and security need to be considered. As the system uses new data to make predictions or take action, both the data and decisions made from the data may be subject to privacy or security concerns.

 - Describe considerations for inclusiveness in an AI solution
   - AI systems should empower everyone and engage people. AI should bring benefits to all parts of society, regardless of physical ability, gender, sexual orientation, ethnicity, or other factors.

 - Describe considerations for transparency in an AI solution
   - AI systems should be understandable. Users should be made fully aware of the purpose of the system, how it works, and what limitations may be expected.

 - Describe considerations for accountability in an AI solution
   - People should be accountable for AI systems. Designers and developers of AI-based solutions should work within a framework of governance and organizational principles that ensure the solution meets ethical and legal standards that are clearly defined.

## Describe fundamental principles of machine learning on Azure (20–25%)
### Identify common machine learning techniques

Machine learning has its origins in statistics and mathematical modeling of data. The fundamental idea of machine learning is to use data from past observations to predict unknown outcomes or values.

 - Identify regression machine learning scenarios: a form of supervised machine learning in which the label predicted by the model is a numeric value.
   - The number of ice creams sold on a given day, based on the temperature, rainfall, and windspeed.
   - The selling price of a property based on its size in square feet, the number of bedrooms it contains, and socio-economic metrics for its location.
   - The fuel efficiency (in miles-per-gallon) of a car based on its engine size, weight, width, height, and length.
 
 - Identify classification machine learning scenarios: Classification is a form of supervised machine learning in which the label represents a categorization, or class. There are two common classification scenarios, Binary classification and Multiclass classification.
   - Whether a patient is at risk for diabetes based on clinical metrics like weight, age, blood glucose level, and so on. (**Binary**)
   - Whether a bank customer will default on a loan based on income, credit history, age, and other factors. (**Binary**)
   - Whether a mailing list customer will respond positively to a marketing offer based on demographic attributes and past purchases. (**Binary**)
   - The species of a penguin (Adelie, Gentoo, or Chinstrap) based on its physical measurements. (**Multiclass**)
   - The genre of a movie (comedy, horror, romance, adventure, or science fiction) based on its cast, director, and budget. (**Multilabel**)

 - Identify clustering machine learning scenarios: The most common form of unsupervised machine learning is clustering. A clustering algorithm identifies similarities between observations based on their features, and groups them into discrete clusters. In some cases, clustering is used to determine the set of classes that exist before training a classification model.
   - Group similar flowers based on their size, number of leaves, and number of petals.
   - Identify groups of similar customers based on demographic attributes and purchasing behavior.

 - Identify features of deep learning techniques: Deep learning is an advanced form of machine learning that tries to emulate the way the human brain learns. The key to deep learning is the creation of an artificial neural network that simulates electrochemical activity in biological neurons by using mathematical functions.
   - made up of multiple layers of neurons - essentially defining a deeply nested function.
   - The weights in a neural network are central to how it calculates predicted values for labels. During the training process, the model learns the weights that will result in the most accurate predictions.
### Describe core machine learning concepts
 - Identify features and labels in a dataset for machine learning

 - Describe how training and validation datasets are used in machine learning
### Describe Azure Machine Learning capabilities
 - Describe capabilities of Automated machine learning

 - Describe data and compute services for data science and machine learning

 - Describe model management and deployment capabilities in Azure Machine Learning


## Describe features of computer vision workloads on Azure (15–20%)
### Identify common types of computer vision solution
 - Identify features of image classification solutions

 - Identify features of object detection solutions

 - Identify features of optical character recognition solutions

 - Identify features of facial detection and facial analysis solutions
### Identify Azure tools and services for computer vision tasks
 - Describe capabilities of the Azure AI Vision service

 - Describe capabilities of the Azure AI Face detection service

 - Describe capabilities of the Azure AI Video Indexer service
## Describe features of Natural Language Processing (NLP) workloads on Azure (15–20%)
### Identify features of common NLP Workload Scenarios
 - Identify features and uses for key phrase extraction

 - Identify features and uses for entity recognition

 - Identify features and uses for sentiment analysis

 - Identify features and uses for language modeling

 - Identify features and uses for speech recognition and synthesis

 - Identify features and uses for translation
### Identify Azure tools and services for NLP workloads
 - Describe capabilities of the Azure AI Language service

 - Describe capabilities of the Azure AI Speech service

 - Describe capabilities of the Azure AI Translator service
## Describe features of generative AI workloads on Azure (15–20%)
### Identify features of generative AI solutions
 - Identify features of generative AI models

 - Identify common scenarios for generative AI

 - Identify responsible AI considerations for generative AI
### Identify capabilities of Azure OpenAI Service
 - Describe natural language generation capabilities of Azure OpenAI Service

 - Describe code generation capabilities of Azure OpenAI Service

 - Describe image generation capabilities of Azure OpenAI Service
