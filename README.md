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
   - **training algorithms:**
     - linear regression.(or some other algorithms)
   - **Evaluation:**
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Coefficient of determination (R2)
     - Area Under the Curve (AUC) -- using received operator characteristic (ROC) curve
       
 - Identify classification machine learning scenarios: Classification is a form of supervised machine learning in which the label represents a categorization, or class. There are two common classification scenarios, Binary classification and Multiclass classification.
   - Whether a patient is at risk for diabetes based on clinical metrics like weight, age, blood glucose level, and so on. (**Binary**)
   - Whether a bank customer will default on a loan based on income, credit history, age, and other factors. (**Binary**)
   - Whether a mailing list customer will respond positively to a marketing offer based on demographic attributes and past purchases. (**Binary**)
   - The species of a penguin (Adelie, Gentoo, or Chinstrap) based on its physical measurements. (**Multiclass**)
   - The genre of a movie (comedy, horror, romance, adventure, or science fiction) based on its cast, director, and budget. (**Multilabel**)
   - **training algorithms(Binary):**
     - logistic regression(S-shaped)
   - **training algorithms(Multiclass):**
     - One-vs-Rest (OvR) algorithms
     - Multinomial algorithms
   - **Evaluation(Binary):**
     - Accuracy
     - Recall
     - Precision
     - F1-score
     - Area Under the Curve (AUC) -- using received operator characteristic (ROC) curve
   - **Evaluation(Multiclass):**
     - overall Accuracy
     - overall Recall
     - overall Precision
     - overall F1-score

 - Identify clustering machine learning scenarios: The most common form of unsupervised machine learning is clustering. A clustering algorithm identifies similarities between observations based on their features, and groups them into discrete clusters. In some cases, clustering is used to determine the set of classes that exist before training a classification model.
   - Group similar flowers based on their size, number of leaves, and number of petals.
   - Identify groups of similar customers based on demographic attributes and purchasing behavior.
   - **Evaluation:** Average distance to cluster center, Average distance to other center, Maximum distance to cluster center, Silhouette.

 - Identify features of deep learning techniques: Deep learning is an advanced form of machine learning that tries to emulate the way the human brain learns. The key to deep learning is the creation of an artificial neural network that simulates electrochemical activity in biological neurons by using mathematical functions.
   - made up of multiple layers of neurons - essentially defining a deeply nested function.
   - The weights in a neural network are central to how it calculates predicted values for labels. During the training process, the model learns the weights that will result in the most accurate predictions.
   - feature vector
   - deep neural networks (DNNs)
   - using a softmax or similar function to calculate the probability distribution
   - A loss function is used to compare the predicted ŷ values to the known y values and aggregate the difference (which is known as the loss)
   - optimization technique can vary, but usually involves a gradient descent approach in which each weight is increased or decreased to minimize the loss.
   - The changes to the weights are backpropagated to the layers in the network, replacing the previously used values.
   - The process is repeated over multiple iterations (known as epochs) until the loss is minimized and the model predicts acceptably accurately.

### Describe core machine learning concepts
 - Identify features and labels in a dataset for machine learning
   -  the observations include the observed attributes or features of the thing being observed
   -  the known value of the thing you want to train a model to predict (known as the label).
     
 - Describe how training and validation datasets are used in machine learning
   - training: an algorithm is applied to the data to try to determine a relationship between the features and the label, and generalize that relationship as a calculation that can be performed on x to calculate y. The whole process is iterative and the parameters of the algorithm are adjusted iteratively.
   - validation: After each training iteration or epoch, the model is evaluated on the validation dataset. The validation performance helps in making decisions such as stopping training when the model starts to overfit or adjusting hyperparameters to improve generalization.
   
### Describe Azure Machine Learning capabilities
 - Describe capabilities of Automated machine learning
   - easy to run multiple training jobs with different algorithms and parameters to find the best model for your data.
 - Describe data and compute services for data science and machine learning
   - Azure Machine Learning
   - Azure Databricks
   - Azure Synapse Analytics
   - Azure Data Factory
   - Azure HDInsight
   - Azure AI Search
 - Describe model management and deployment capabilities in Azure Machine Learning
   - Model Registration
   - Model Deployment
   - Environment Management
   - Automated Machine Learning (AutoML) Deployment
   - Model Monitoring and Logging
   - Scalability and Load Balancing
   - Integration with Azure DevOps
   - Role-Based Access Control (RBAC)
   - Continuous Integration and Continuous Deployment (CI/CD)

## Describe features of computer vision workloads on Azure (15–20%)
### Identify common types of computer vision solution
 - **Key words:** kernels,filter, convolutional filtering, convolutional neural network (CNN), softmax, loss, Transformers, Multi-modal models.
 - Identify features of image classification solutions: categorize images into predefined classes or labels.
   - Convolutional Neural Networks (CNNs)
   - Transfer Learning
   - Multi-Class Classification
   - Model Monitoring and Maintenance
   - Support for Custom Models
 - Identify features of object detection solutions: identify and locate objects within an image or video frame.
   - Convolutional Neural Networks (CNNs)
   - Region Proposal Networks (RPN)
   - Bounding Box Regression
   - Multi-Class Detection
   - Transfer Learning
   - Robustness to Occlusion (partially visible)
   - Handling Multiple Object Instances
 - Identify features of optical character recognition solutions
   - Text Extraction
   - Document Layout Analysis
   - Handwritten Text Recognition
   - Printed Text Recognition
   - Multi-language Support
   - Table and Form Extraction
   - Accuracy and Precision
   - Character Segmentation
   - Customizable and Trainable Models
   - Security and Privacy

 - Identify features of facial detection and facial analysis solutions
   - **facial detection**
     - Face Detection
     - Real-time Detection
     - Multi-face Detection
     - Accuracy and Precision
     - Scale and Pose Variation
     - Illumination and Lighting Robustness
     - Age and Gender Estimation
   - **facial analysis**
     - Expression Analysis
     - Facial Recognition
     - Facial Similarity Matching
     - Facial Attribute Analysis
     - Facial Biometrics
     - Privacy Protection Measures
     - Anti-spoofing Measures
### Identify Azure tools and services for computer vision tasks
 - Describe capabilities of the Azure AI Vision service
   - Optical character recognition (OCR) - extracting text from images.
   - Generating captions and descriptions of images.
   - Detection of thousands of common objects in images.
   - Tagging visual features in images

 - Describe capabilities of the Azure AI Face detection service
   - Face Detection
   - Face Identification
   - Face Verification
   - Age and Gender Estimation
   - Glasses Detection
   - Facial Landmarks  

 - Describe capabilities of the Azure AI Video Indexer service:  analyzing and extracting insights from videos.
   - Automatic Transcription
   - Face Detection and Recognition
   - Object and Scene Detection
   - Visual Content Moderation
   - OCR (Optical Character Recognition)
   - Custom Labels and Models
   - Integration with Azure Services
   - Sentiment Analysis

## Describe features of Natural Language Processing (NLP) workloads on Azure (15–20%)
### keywords
 - Stemming
 - n-gram
 - normalization
 - Stopword removal 
### Identify features of common NLP Workload Scenarios
 - Identify features and uses for key phrase extraction: identifying and extracting the most important phrases or terms from a text.
   - Identifying Important Terms, Frequency Analysis, Summarization Assistance, Multi-lingual Support
   - Document Summarization, Content Tagging, Content Recommendation, Sentiment Analysis
 - Identify features and uses for entity recognition: identifying and classifying entities (such as names of people, organizations, locations, dates, etc.) in a text.
   - Contextual Understanding, Recognition of Various Entity Types, Entity Linking
   - Information Extraction, Question Answering Systems, Sentiment Analysis, Chatbots and Virtual Assistants, Named Entity Disambiguation

 - Identify features and uses for sentiment analysis: determining the sentiment expressed in a piece of text—whether it's positive, negative, or neutral.
   - Binary and Multiclass Sentiment Classification, Emotion Detection, Context Awareness, Opinion Summarization

 - Identify features and uses for language modeling: predicting the probability of a sequence of words or characters in a given context.
   - Contextual Embeddings, Generative Text Completion, Zero-shot Learning, Multimodal Language Models, Bidirectional Modeling, Probabilistic Predictions
   - Text Generation, Question Answering, Summarization, Spell Checking and Auto-correction, Conversational Agents and Chatbots

 - Identify features and uses for speech recognition and synthesis: deal with understanding and generating spoken language.
   - **Speech Recognition**:
     - The model that is used by the Speech to text API, is based on the **Universal Language Model** that was trained by Microsoft. The data for the model is Microsoft-owned and deployed to Microsoft Azure. The model is **optimized** for two scenarios, **conversational and dictation**. You can also create and train your own custom models including acoustics, language, and pronunciation if the pre-built models from Microsoft do not provide what you need.
     - **acoustic model:** converts the audio signal into phonemes (representations of specific sounds).
     - **language model:** maps phonemes to words, usually using a statistical algorithm that predicts the most probable sequence of words based on the phonemes.
     - Automatic Speech Recognition (ASR), Speaker Identification, Noise Reduction, Language Adaptation, Keyword Spotting
     - Voice Assistants, Transcription Services, Voice Search, Interactive Voice Response (IVR) Systems
   - **Speech Synthesis**:
     - Text-to-Speech, Emotional and Expressive Synthesis, Voice Cloning, Pitch and Rate Control
     - Voiceovers for Media Content, Interactive Learning Applications, Notification Systems, Entertainment and Gaming

 - Identify features and uses for translation: converting text from one language to another.
   - Multilingual Support, Bidirectional Translation, Speech-to-Text Translation
   - Global Content Localization, Language Learning, Collaboration and Knowledge Sharing, News and Media Translation
   
### Identify Azure tools and services for NLP workloads
 - Describe capabilities of the Azure AI Language service
   - Entity recognition
   - Entity linking
   - Personal identifying information (PII) detection
   - Language detection
   - Sentiment analysis and opinion mining
   - Key phrase extraction
   - Summarization

 - Describe capabilities of the Azure AI Speech service
   - The Speech to text API: Real-time transcription, Batch transcription
   - The Text to speech API: Speech synthesis voices
   - both APIs support a variety of languages

 - Describe capabilities of the Azure AI Translator service
   - Text Translation
   - Speech Translation
   - Language Detection
   - Batch Translation
   - Document Translation
   - Azure AI Search Integration

## Describe features of generative AI workloads on Azure (15–20%)
### Identify features of generative AI solutions
 - Identify features of generative AI models:  powered by large language models (LLMs). Take in natural language input, and return appropriate responses in a variety of formats such as natural language, images, or code.
   - Generative Capabilities
   - Unsupervised Learning
   - Transfer Learning
   - Variability and Creativity
   - Conditional Generation
   - Autoencoders and Latent Representations
   - GANs (Generative Adversarial Networks)
   - Variational Autoencoders (VAEs)
   - Text Generation
   - Image-to-Image Translation
   
 - Identify common scenarios for generative AI
   - Natural language generation
   - Image generation
   - Code generation

 - Identify responsible AI considerations for generative AI: Plan a responsible generative AI solution
   - Identify potential harms that are relevant to your planned solution.
     - Identify potential harms
     - Prioritize identified harms: assess the likelihood of its occurrence and the resulting level of impact if it does.
     - Test and verify the prioritized harms (red teaming)
     - Document and share the verified harms
   - Measure the presence of these harms in the outputs generated by your solution.
     - Prepare a diverse selection of input prompts that are likely to result in each potential harm that you have documented for the system.
     - Submit the prompts to the system and retrieve the generated output.
     - Apply pre-defined criteria to evaluate the output and categorize it according to the level of potential harm it contains.
   - Mitigate the harms at multiple layers in your solution to minimize their presence and impact, and ensure transparent communication about potential risks to users.
     - Model: 1)Selecting a model that is appropriate for the intended solution use. 2)Fine-tuning a foundational model with your own training data so that the responses it generates are more likely to be relevant and scoped to your solution scenario.
     - Safety System: 1)platform-level configurations and capabilities that help mitigate harm(Azure OpenAI Service: content filter). 2)abuse detection algorithms to determine if the solution is being systematically abused (for example through high volumes of automated requests from a bot) and alert notifications that enable a fast response to potential system abuse or harmful behavior.
     - Metaprompt and grounding: focuses on the construction of prompts that are submitted to the model.
       - Specifying metaprompts or system inputs that define behavioral parameters for the model.
       - Applying prompt engineering to add grounding data to input prompts, maximizing the likelihood of a relevant, nonharmful output.
       - Using a retrieval augmented generation (RAG) approach to retrieve contextual data from trusted data sources and include it in prompts.
     - User experience
       - Designing the application user interface to constrain inputs to specific subjects or types, or applying input and output validation can mitigate the risk of potentially harmful responses.
       - Documentation and other descriptions of a generative AI solution should be appropriately transparent about the capabilities and limitations of the system, the models on which it's based, and any potential harms that may not always be addressed by the mitigation measures you have put in place.
   - Operate the solution responsibly by defining and following a deployment and operational readiness plan.
     - prerelease reviews:
       - Legal
       - Privacy
       - Security
       - Accessibility
     - Release and operate:
       - Devise a phased delivery plan
       - Create an incident response plan
       - Create a rollback plan
       - Implement the capability to immediately block harmful system responses when they're discovered.
       - Implement a capability to block specific users, applications, or client IP addresses in the event of system misuse.
       - Implement a way for users to provide feedback and report issues.
       - Track telemetry data that enables you to determine user satisfaction and identify functional gaps or usability challenges. Telemetry collected should comply with privacy laws and your own organization's policies and commitments to user privacy.

### Identify capabilities of Azure OpenAI Service
 - Describe natural language generation capabilities of Azure OpenAI Service
   - Azure OpenAI's natural language models are able to take in natural language and are excellent at both understanding and creating natural language.

 - Describe code generation capabilities of Azure OpenAI Service
   - GPT models are able to take natural language or code snippets and translate them into code. The OpenAI GPT models are proficient in over a dozen languages, such as C#, JavaScript, Perl, PHP, and is most capable in Python. GPT models have been trained on both natural language and billions of lines of code from public repositories. The models are able to generate code from natural language instructions such as code comments, and can suggest ways to complete code functions.

 - Describe image generation capabilities of Azure OpenAI Service
   - Image generation models can take a prompt, a base image, or both, and create something new. These generative AI models can create both realistic and artistic images, change the layout or style of an image, and create variations on a provided image.

## Describe features of Document Intelligence and Knowledge Mining workloads on Azure
### Identify capabilities of Document Intelligence
- Describe capabilities of document intelligence
  - extract text, layout, and key-value pairs are known as document analysis. Document analysis provides locations of text on a page identified by bounding box coordinates.

- Identify features of Azure AI Document Intelligence
  - Prebuilt models - pretrained models that have been built to process common document types such as invoices, business cards, ID documents, and more. These models are designed to recognize and extract specific fields that are important for each document type.
  - Custom models - can be trained to identify specific fields that are not included in the existing pretrained models.
  - Document analysis - general document analysis that returns structured data representations, including regions of interest and their inter-relationships.

### Identify features of Knowledge Mining workloads and Azure AI Search
- Describe features of Azure AI Search: provides the infrastructure and tools to create search solutions that extract data from various structured, semi-structured, and non-structured documents.
  - Data from any source: accepts data from any source provided in JSON format, with auto crawling support for selected data sources in Azure.
  - Full text search and analysis: offers full text search capabilities supporting both simple query and full Lucene query syntax.
  - AI powered search: has Azure AI capabilities built in for image and text analysis from raw content.
  - Multi-lingual offers linguistic analysis for 56 languages to intelligently handle phonetic matching or language-specific linguistics. Natural language processors 
available in Azure AI Search are also used by Bing and Office.
  - Geo-enabled: supports geo-search filtering based on proximity to a physical location.
  - Configurable user experience: has several features to improve the user experience including autocomplete, autosuggest, pagination, and hit highlighting.
