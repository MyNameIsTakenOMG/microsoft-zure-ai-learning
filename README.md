# Master the basics of Azure: AI Fundamentals (AI-900)
## Learning Path 1. Microsoft Azure AI Fundamentals: AI Overview 
### Module 1. Fundamental AI Concepts
####  AI is software that imitates human behaviors and capabilities. Key workloads include:
  - **Machine learning** - This is often the foundation for an AI system, and is the way we "teach" a computer model to make predictions and draw conclusions from data.
  - **Computer vision** - Capabilities within AI to interpret the world visually through cameras, video, and images.
  - **Natural language processing** - Capabilities within AI for a computer to interpret written or spoken language, and respond in kind.
  - **Document intelligence** - Capabilities within AI that deal with managing, processing, and using high volumes of data found in forms and documents.
  - **Knowledge mining** - Capabilities within AI to extract information from large volumes of often unstructured data to create a searchable knowledge store.
  - **Generative AI** - Capabilities within AI that create original content in a variety of formats including natural language, image, code, and more.
#### How machine learning works
  **Machine learning models try to capture the relationship between data.**
  - Collecting data on samples.
  - labeling the samples with the correct values.
  - The labeled data is processed using an algorithm that finds relationships between the features of the samples and the labeled values.
  - The results of the algorithm are encapsulated in a model.
  - When new samples are found, the model can identify the correct label.
---
**Note:** **Azure Machine Learning service** - a cloud-based platform for creating, managing, and publishing machine learning models. **Azure Machine Learning Studio** offers multiple authoring experiences such as:
 - Automated machine learning: this feature enables non-experts to quickly create an effective machine learning model from data.
 - Azure Machine Learning Designer: a graphical interface enabling no-code development of machine learning solutions.
 - Data metric visualization: analyze and optimize your experiments with visualization.
 - Notebooks: write and run your own code in managed Jupyter Notebook servers that are directly integrated in the studio.
#### Computer Vision models and capabilities
  **Common computer vision tasks:**
   - Image classification
   - Object detection
   - Semantic segmentation
   - Image analysis
   - Face detection, analysis, and recognition
   - Optical character recognition (OCR)
---
**Note:** **Microsoft's Azure AI Vision**
 - Image Analysis: capabilities for analyzing images and video, and extracting descriptions, tags, objects, and text.
 - Face: capabilities that enable you to build face detection and facial recognition solutions.
 - Optical Character Recognition (OCR): capabilities for extracting printed or handwritten text from images, enabling access to a digital version of the scanned text.

#### Natural language processing (NLP)
**Natural language processing (NLP)** is the area of AI that deals with creating software that understands written and spoken language.
NLP enables you to create software that can:
 - Analyze and interpret text in documents, email messages, and other sources.
 - Interpret spoken language, and synthesize speech responses.
 - Automatically translate spoken or written phrases between languages.
 - Interpret commands and determine appropriate actions.
**Note:** **Microsoft's Azure AI Language** to build natural language processing solutions. Some features of Azure AI Language include understanding and analyzing text, training conversational language models that can understand spoken or text-based commands, and building intelligent applications. **Microsoft's Azure AI Speech** is another service that can be used to build natural language processing solutions. Azure AI Speech features include speech recognition and synthesis, real-time translations, conversation transcriptions, and more.

#### Document Intelligence and Knowledge Mining
**Document Intelligence** is the area of AI that deals with managing, processing, and using high volumes of a variety of data found in forms and documents. Document intelligence enables you to create software that can automate processing for contracts, health documents, financial forms and more.
**Note:** You can use **Microsoft's Azure AI Document Intelligence** to build solutions that manage and accelerate data collection from scanned documents. Features of Azure AI Document Intelligence help automate document processing in applications and workflows, enhance data-driven strategies, and enrich document search capabilities.
**Knowledge mining** is the term used to describe solutions that involve extracting information from large volumes of often unstructured data to create a searchable knowledge store.
**Note:** One Microsoft knowledge mining solution is **Azure AI Search**, a private, enterprise, search solution that has tools for building indexes. The indexes can then be used for internal only use, or to enable searchable content on public facing internet assets.

#### Generative AI
**Generative AI** describes a category of capabilities within AI that create original content. People typically interact with generative AI that has been built into chat applications. Generative AI applications take in natural language input, and return appropriate responses in a variety of formats including natural language, image, code, and audio.
**Note:** In Microsoft Azure, you can use the **Azure OpenAI** service to build generative AI solutions. Azure OpenAI Service is Microsoft's cloud solution for deploying, customizing, and hosting generative AI models. It brings together the best of OpenAI's cutting edge models and APIs with the security and scalability of the Azure cloud platform.

#### Challenges and risks with AI
**potential challenges and risks:**
 - Bias can affect results
 - Errors may cause harm
 - Data could be exposed
 - Solutions may not work for everyone
 - Users must trust a complex system
 - Who's liable for AI-driven decisions?

#### Understand Responsible AI
**Six principles:**
 - Fairness: AI systems should treat all people fairly.
 - Reliability and safety: AI systems should perform reliably and safely.
 - Privacy and security: AI systems should be secure and respect privacy.
 - Inclusiveness: AI systems should empower everyone and engage people. AI should bring benefits to all parts of society, regardless of physical ability, gender, sexual orientation, ethnicity, or other factors.
 - Transparency: AI systems should be understandable. Users should be made fully aware of the purpose of the system, how it works, and what limitations may be expected.
 - Accountability: People should be accountable for AI systems. Designers and developers of AI-based solutions should work within a framework of governance and organizational principles that ensure the solution meets ethical and legal standards that are clearly defined.

### Module 2. Fundamentals of machine learning
#### Machine learning as a function
Fundamentally, a machine learning model is a software application that encapsulates a function to calculate an output value based on one or more input values. The process of defining that function is known as **training**. After the function has been defined, you can use it to predict new values in a process called **inferencing**.

#### Types of machine learning
- **Supervised machine learning**: Supervised machine learning is a general term for machine learning algorithms in which the training data includes both feature values and known label values.
  - **Regression**: a form of supervised machine learning in which the label predicted by the model is a numeric value.
  - **Classification**: a form of supervised machine learning in which the label represents a categorization, or class.
    - Binary classification: the label determines whether the observed item is (or isn't) an instance of a specific class.
    - Multiclass classification: Multiclass classification extends binary classification to predict a label that represents one of multiple possible classes.
- **Unsupervised machine learning**: Unsupervised machine learning involves training models using data that consists only of feature values without any known labels. Unsupervised machine learning algorithms determine relationships between the features of the observations in the training data.
  - Clustering:  A clustering algorithm identifies similarities between observations based on their features, and groups them into discrete clusters. In some cases, clustering is used to determine the set of classes that exist before training a classification model.
