# Curriculum Compass: A Hybrid RAG Chatbot Empowering Northeastern Students in Course Selection

## Table of Contents
- [Overview](#overview)
- [Data Collection](#data-collection)
  - [Review Data](#review-data)
    - [Review Data Schema](#review-data-schema)
  - [Course Data](#course-data)
    - [Course Data Schema](#course-data-schema)
- [Query Validation](#query-validation)
  - [First-Level Validation (LLMGuard)](#first-level-validation-llmguard)
  - [Second-Level Validation (Relevancy-Test)](#second-level-validation-relevancy-test)
- [Hybrid RAG: Integrated Retrieval-Augmented Generation (RAG) System for Curriculum Compass](#hybrid-rag-integrated-retrieval-augmented-generation-rag-system-for-curriculum-compass)
  - [Hybrid Retrieval for Course Information](#hybrid-retrieval-for-course-information)
    - [Tabular Retrieval Using TF-IDF](#tabular-retrieval-using-tf-idf)
    - [Dense Embedding Retrieval](#dense-embedding-retrieval)
    - [Re-ranking with a Cross-Encoder](#re-ranking-with-a-cross-encoder)
  - [Dense Retrieval for Reviews](#dense-retrieval-for-reviews)
  - [Integrated RAG Pipeline](#integrated-rag-pipeline)
    - [Parallel Retrieval](#parallel-retrieval)
    - [Combining and Re-ranking](#combining-and-re-ranking)
    - [Response Generation](#response-generation)
    - [System Design and Workflow](#system-design-and-workflow)
    - [Initial Design Summary](#initial-design-summary)
    - [Conclusion (Hybrid RAG)](#conclusion-hybrid-rag)
- [Synthetic Dataset Creation](#synthetic-dataset-creation)
  - [Dataset Creation Process](#dataset-creation-process)
    - [Seed Queries](#seed-queries)
    - [Rephrased Variants](#rephrased-variants)
    - [Context Retrieval](#context-retrieval)
    - [Response Generation](#response-generation-1)
    - [Topics and Course Coverage](#topics-and-course-coverage)
    - [Dataset Composition](#dataset-composition)
    - [Dataset Repository](#dataset-repository)
    - [Conclusion (Dataset)](#conclusion-dataset)
- [Evaluation](#evaluation)
- [License](#license)
- [Contact](#contact)


## Overview
**Curriculum Compass** is a Hybrid Retrieval-Augmented Generation (RAG) chatbot designed to assist Northeastern University students in selecting courses. By combining historical course review data from NuTrace with upcoming course offerings from the NU Banner API, the chatbot aims to provide personalized guidance and insights to help students make informed decisions.

## Data Collection

### Review Data
We collected previous course reviews from **NuTrace**. These reviews come in the form of instructor report files containing feedback from students about various aspects of the course and the instructor’s teaching.

To process these reports:
1. We used **PyMuPDF** to parse PDF documents.
2. We extracted student feedback for questions such as:
   - *What were the strengths of this course and/or this instructor?*
   - *What could the instructor do to make this course better?*
   - *Please expand on the instructor’s strengths and/or areas for improvement in facilitating inclusive learning.*
   - *Please comment on your experience of the online course environment in the open-ended text box.*
   - *What I could have done to make this course better for myself.*

These reviews are then used to augment the Large Language Model (LLM) with knowledge about previous course offerings, enabling it to answer student queries more effectively.

#### Review Data Schema

```json
{
  "type": "object",
  "properties": {
    "CRN": {
      "type": "string",
      "description": "The unique Course Reference Number (CRN) for the course."
    },
    "Course Name": {
      "type": "string",
      "description": "The name of the course."
    },
    "Instructor": {
      "type": "string",
      "description": "The name of the instructor teaching the course."
    },
    "Subject": {
      "type": "string",
      "description": "The subject or discipline of the course (e.g., 'Computer Science')."
    },
    "Course Number": {
      "type": "string",
      "description": "The specific course number associated with the subject."
    },
    "Question": {
      "type": "string",
      "description": "The question related to the course, as part of the review data."
    },
    "Review": {
      "type": "string",
      "description": "The student's review or feedback related to the question."
    }
  },
  "required": [
    "CRN",
    "Course Name",
    "Instructor",
    "Subject",
    "Course Number",
    "Question",
    "Review"
  ],
  "additionalProperties": false
}
```

### Course Data
Next, we obtained the upcoming semester’s course offerings using the [NU Banner API](https://jennydaman.gitlab.io/nubanned/dark.html). This data helps provide students with the most up-to-date information on what courses are available, along with details such as the instructor’s name, class schedule, and prerequisites.

#### Course Data Schema

```json
{
  "type": "object",
  "properties": {
    "CRN": {
      "type": "string",
      "description": "The unique Course Reference Number (CRN) for the course."
    },
    "Campus Description": {
      "type": "string",
      "description": "The description of the campus where the course is offered."
    },
    "Course Title": {
      "type": "string",
      "description": "The title of the course."
    },
    "Subject Course": {
      "type": "string",
      "description": "The subject and course identifier (e.g., 'CS5100')."
    },
    "Faculty Name": {
      "type": "string",
      "description": "The name of the faculty member teaching the course."
    },
    "Course Description": {
      "type": "string",
      "description": "A detailed description of the course content."
    },
    "Term": {
      "type": "string",
      "description": "The academic term during which the course is offered (e.g., 'Spring 2025')."
    },
    "Begin Time": {
      "type": "string",
      "description": "The start time of the course in HH:MM format."
    },
    "End Time": {
      "type": "string",
      "description": "The end time of the course in HH:MM format."
    },
    "Days": {
      "type": "string",
      "description": "The days of the week when the course is scheduled (e.g., 'MWF')."
    },
    "Prerequisites": {
      "type": "string",
      "description": "A list or description of the prerequisites for the course."
    }
  },
  "required": [
    "CRN",
    "Campus Description",
    "Course Title",
    "Subject Course",
    "Faculty Name",
    "Course Description",
    "Term",
    "Begin Time",
    "End Time",
    "Days",
    "Prerequisites"
  ],
  "additionalProperties": false
}
```

### Query Validation

This component is responsible for verifying that user queries meet specific policy standards and are sufficiently relevant before they proceed in the system. It employs a two-level validation approach, combining both an LLM-based guardrail system and a relevancy check.


**1. First-Level Validation (LLMGuard)**

- **Checks for queries that violate any of the following:**
  - Gibberish or nonsensical content  
  - Prompt injection attempts  
  - Banned substrings  
  - Exceeding the token limit  

- **If a violation is detected:**
  - Generate a user-friendly explanation describing:
    - The nature of the violation  
    - A suggested rephrasing to comply with the content policy  

- **If no violation is detected:**
  - Proceed to relevancy checks  

**2. Second-Level Validation (Relevancy Test)**

- **Assesses whether the query relates to Northeastern University (NEU) courses or professors.**
- **Uses an LLM prompt** to decide if the query is “RELEVANT” or “NOT RELEVANT.”

- **If marked “NOT RELEVANT”:**
  - Generate a user-friendly explanation  
  - Provide a suggested topic or question aligned with NEU academics  

- **If relevant:**
  - The query is allowed to move forward in the pipeline  

#### First-Level Validation: LLMGuard

To ensure queries meet fundamental content requirements, we leverage a custom class called **LLMGuard**. This class internally uses an LLM-based guardrail library to detect various forms of violations.

##### Checks Performed

1. **Gibberish**  
   Determines if the query is largely nonsensical or unintelligible.

2. **Banned Substrings**  
   Looks for the presence of certain disallowed words or phrases (e.g., explicit content, hateful language).

3. **Prompt Injection**  
   Identifies manipulative instructions attempting to override or disrupt system behavior.

4. **Token Limit Exceeded**  
   Verifies the query stays within a defined size limit for processing.

##### Failure Response

If any of these checks fail, the system automatically generates a concise, user-friendly explanation using an LLM prompt. The explanation:

- Acknowledges the content policy violation.
- Briefly and constructively explains why the query was flagged.
- Suggests a revised query that aligns with acceptable language and content guidelines (focusing on academic and professional phrasing).

##### Prompt Example

Below is an illustrative snippet of the prompt used to generate user-friendly responses for guard failures:

```python
system_prompt = (
    "You are an AI designed to provide information on Northeastern University courses and professors. "
    "For off-topic queries, respond with exactly three lines:\n"
    "1. First line must be exactly: 'NOT RELEVANT'\n"
    "2. Second line: explanation properly why the query is irrelavent \n"
    "3. Third line: 'Suggested question: ' followed by a question about NEU courses/professors. "
    "Questions should vary between these types:\n"
    "   - Course content questions (e.g., 'What topics are covered in NEU's Machine Learning course?')\n"
    "   - Teaching style questions (e.g., 'How does Professor X teach Database Management?')\n"
    "   - Course reviews/experience (e.g., 'How challenging is the Algorithms course at NEU?')\n"
    "   - Course structure (e.g., 'What projects are included in Software Engineering?')\n"
    "   - Professor expertise (e.g., 'Which professors specialize in AI at NEU?')\n"
    "\nFocus on these CS topics:\n"
    "   - Machine Learning\n"
    "   - Algorithms\n"
    "   - Database Management Systems\n"
    "   - Artificial Intelligence\n"
    "   - Data Structures\n"
    "   - Software Engineering\n"
    "   - Computer Networks\n"
    "   - Operating Systems"
)

user_message = f"""
User query: {query}

Instructions:
Generate a 3-line response with:
Line 1: 'NOT RELEVANT'
Line 2: Explain why this query isn't about NEU academics
Line 3: Suggest a question about NEU courses/professors that covers either:
    - Course content and topics
    - Teaching methods and style
    - Student experiences and reviews
    - Course structure and assignments
    - Professor expertise and approach
Make the suggestion feel natural and focused on what students might want to know.
"""
```

### Second-Level Validation: Relevancy Test

Even if a query passes the initial guardrail checks, it might still be unrelated to NEU courses or professors. This test ensures only relevant queries proceed.


#### Checks Performed

1. **Relevancy Prompt**  
   The system uses a well-structured prompt that outlines clear relevance criteria for NEU-related queries. Specifically, relevant queries must directly address topics such as:
   - Course offerings, schedules, prerequisites, or location  
   - Professor information (e.g., teaching style, faculty background)  
   - Course or professor reviews  
   - Any content clearly tied to NEU academic inquiries  

2. **LLM Response**  
   - If the query is about NEU courses/professors, the LLM responds with **"RELEVANT"**.  
   - Otherwise, the LLM responds with **"NOT RELEVANT"**.

Below is an illustrative snippet of the prompt used for evaluating the relevance of the input query:

```python
system_message = """
You are a helpful AI system tasked with filtering user questions about Northeastern University courses, professors, and course reviews.

### Relevancy Rules
- Relevant questions are those about:
  • Course offerings, schedules, prerequisites, or location (campus vs. online).
  • Professor/faculty information (e.g., who is teaching, professor's teaching style).
  • Opinions or reviews about the course or professor (e.g., workload, grading difficulty).
  • Past or present course reviews (e.g., "Has this course been offered in the past? How were the reviews?").
  • Anything else directly related to Northeastern courses or professors.

- Irrelevant questions:
  • Topics unrelated to Northeastern courses or professors (e.g., weather, jokes, cooking).
  • Personal advice not connected to Northeastern's courses/professors.
  • Any query that does not pertain to course data or professor data at Northeastern.

Your output:
- Respond EXACTLY with 'RELEVANT' if the question is about Northeastern courses, professors, or reviews (including workload, grading, difficulty).
- Respond EXACTLY with 'NOT RELEVANT' if it is off-topic.

### Examples
1) User query: "Which professor is teaching CS1800 next semester?"
   Answer: RELEVANT
2) User query: "How do I bake a chocolate cake?"
   Answer: NOT RELEVANT
3) User query: "How much workload does CS1800 typically have?"
   Answer: RELEVANT
4) User query: "How is Professor Karl Lieberherr in terms of grading?"
   Answer: RELEVANT
5) User query: "What is the capital of France?"
   Answer: NOT RELEVANT
6) User query: "Has Data Structures been offered previously? Any reviews about difficulty?"
   Answer: RELEVANT
"""

user_query = f" Query: {{query}}\\n\\nAnswer:"
```

## Failure Response

Queries deemed **"NOT RELEVANT"** trigger an LLM-based explanation. The user receives:

1. A clear statement: **"NOT RELEVANT"**  
2. A brief explanation of why their question does not pertain to NEU academics  
3. A suggested question that is academically focused and related to NEU courses or professors  

### Prompt Example

Below is an illustrative snippet of the prompt used for generating user-friendly responses to irrelevant queries:

```python
system_message = (
    "You are an AI designed to provide information on Northeastern University courses and professors. "
    "For off-topic queries, respond with exactly three lines:\n"
    "1. First line must be exactly: 'NOT RELEVANT'\n"
    "2. Second line: explanation properly why the query is irrelavent \n"
    "3. Third line: 'Suggested question: ' followed by a question about NEU courses/professors. "
    "Questions should vary between these types:\n"
    "   - Course content questions (e.g., 'What topics are covered in NEU's Machine Learning course?')\n"
    "   - Teaching style questions (e.g., 'How does Professor X teach Database Management?')\n"
    "   - Course reviews/experience (e.g., 'How challenging is the Algorithms course at NEU?')\n"
    "   - Course structure (e.g., 'What projects are included in Software Engineering?')\n"
    "   - Professor expertise (e.g., 'Which professors specialize in AI at NEU?')\n"
    "\nFocus on these CS topics:\n"
    "   - Machine Learning\n"
    "   - Algorithms\n"
    "   - Database Management Systems\n"
    "   - Artificial Intelligence\n"
    "   - Data Structures\n"
    "   - Software Engineering\n"
    "   - Computer Networks\n"
    "   - Operating Systems"
)

user_message = f"""
User query: {{query}}

Instructions:
Generate a 3-line response with:
Line 1: 'NOT RELEVANT'
Line 2: Explain why this query isn't about NEU academics
Line 3: Suggest a question about NEU courses/professors that covers either:
    - Course content and topics
    - Teaching methods and style
    - Student experiences and reviews
    - Course structure and assignments
    - Professor expertise and approach
Make the suggestion feel natural and focused on what students might want to know.
"""
```

### The Final Workflow

1. **User Query Received**  
   - The pipeline first passes the user query to **LLMGuard** for content compliance checks.

2. **LLMGuard Validation**  
   - **If the query fails**:  
     - The system immediately returns a user-friendly violation explanation.
   - **If the query passes**:  
     - Proceed to the **Relevancy Test**.

3. **Relevancy Test**  
   - The system checks if the query is sufficiently related to NEU courses/professors.
   - **If deemed irrelevant**:  
     - The system returns a brief explanation and suggests an NEU-related question.
   - **If relevant**:  
     - The query is considered valid and passed on to subsequent components in the pipeline.

## Hybrid RAG: Integrated Retrieval-Augmented Generation (RAG) System for Curriculum Compass

The **Curriculum Compass** project leverages an integrated Retrieval-Augmented Generation (RAG) approach to provide Northeastern University students with accurate and contextual responses regarding course offerings and reviews for the **Spring 2025** semester. The system is built using multiple retrievers and a reranking mechanism, ensuring robust handling of diverse queries and seamless integration of course and review information into a single, cohesive response.

### Hybrid Retrieval for Course Information

The first component, **CourseRAGPipeline**, is responsible for retrieving course-related information using a hybrid retrieval approach:

#### Tabular Retrieval Using TF-IDF
A TF-IDF-based search system retrieves course information by matching query terms with keywords in course descriptions. The **CourseSearchSystem** preprocesses user queries to extract metadata (e.g., course title, professor, term, or campus) and augments the query to improve retrieval accuracy.

#### Dense Embedding Retrieval
Complementing the TF-IDF method, **dense embeddings** enable semantic search. These embeddings capture nuanced meanings of user queries, allowing the system to handle incomplete or ambiguous inputs (for example, interpreting “PDP” for “Programming Design Paradigm”).

#### Re-ranking with a Cross-Encoder
After retrieval, a **Cross-Encoder** model re-ranks the combined results, prioritizing those most relevant to the user's query. This ensures high-quality responses, even when user input contains typos or shorthand phrases.

### Dense Retrieval for Reviews

The second pipeline, **ReviewsRAGPipeline**, is designed to retrieve and rank student reviews of courses. Unlike the CourseRAGPipeline, this system uses **dense embeddings** exclusively. It converts queries and reviews into vector representations and performs similarity searches using a **ChromaDB** database. The retrieved reviews are re-ranked using the same Cross-Encoder model for consistency and quality.

### Integrated RAG Pipeline

The final stage of the system, **IntegratedRAGPipeline**, combines course and review retrievals to provide a comprehensive response.

#### Parallel Retrieval
The **CourseRAGPipeline** and **ReviewsRAGPipeline** process user queries in parallel, retrieving the top-*k* most relevant documents from each source.

#### Combining and Re-ranking
Retrieved course descriptions and reviews are merged into a single document set, with explicit labels (such as `[COURSE INFO]` or `[STUDENT REVIEW]`). A second reranking step ensures the combined results align closely with the query’s intent.

#### Response Generation
The reranked context is passed to a **large language model (LLM)** for natural language response generation. The final response seamlessly integrates course information and student reviews, providing students with actionable insights.

#### System Design and Workflow

The entire system is designed to handle diverse and ambiguous user queries effectively. It employs **preprocessing techniques** to extract structured information from queries, such as course titles, professors, and terms. Queries are enhanced with metadata to improve retrieval accuracy.

By integrating **TF-IDF for keyword-based search** and **dense embeddings for semantic understanding**, the system achieves robust performance across different query types. **Cross-encoder-based reranking** at multiple stages further refines the results, while error-handling mechanisms provide fallback options to maintain system reliability.

The main function initializes all components, including a **Query Validator**, **ChromaDB client**, and individual **RAG pipelines**. The **Query Validator** checks query validity, while the pipelines retrieve and process information to produce responses.

#### Initial Design Summary

For this project, we employ **three distinct retrievers** to handle various types of information:

1. **Course-related information** is retrieved using a **hybrid retriever** that combines:
   - Tabular information retrieval with TF-IDF embeddings  
   - Course description–based retrieval using dense embeddings  

   These results are merged and re-ranked using a **Cross-Encoder** model.

2. **Reviews** are retrieved using a **dense retriever** that locates relevant feedback.

The outputs from the course and review retrievers are then combined and re-ranked again. Finally, the refined context—derived from both retrieval processes—is passed to an **LLM** to generate user-friendly responses that effectively address the user’s query.

#### Conclusion (Hybrid RAG)

The **Curriculum Compass** project demonstrates an advanced application of retrieval and generation techniques to meet the needs of university students. By combining **hybrid retrieval methods**, **dense embeddings**, and **cross-encoder reranking**, the system ensures high-quality responses to even the most ambiguous queries. The integration of multiple pipelines into a unified framework underscores its potential for robust performance and adaptability in real-world applications.

### Synthetic Dataset Creation

**Objective:**  
The goal of this effort was to validate the hypothesis that the capacity of a larger language model (LLM) can be transferred to a smaller LLM up to a certain threshold for domain-specific tasks. To achieve this, we created a synthetic dataset comprising *(question, context, response)* triples. The dataset was generated using a larger LLM, and the smaller LLM was fine-tuned on it to test the hypothesis.


### Dataset Creation Process

#### Seed Queries
- A curated list of seed queries was designed to cover a wide range of student inquiries for the Spring 2025 semester.  
- The seed queries span diverse topics including Machine Learning, Deep Learning, Data Science, Computer Systems, Programming, and Development.

**Examples of seed queries:**
- "What courses are being offered for the {semester} semester?"
- "Which professor is taking the course {course_name}?"
- "What are the courses available related to {topic}?"
- "Can you suggest courses which are not too hectic and easy to get good grades?"

#### Rephrased Variants
- For each seed query, **10 rephrased variants** were generated to ensure diversity in tone and style while maintaining the original semantic meaning.
- Rephrased queries were created using a custom system prompt:

```plaintext
Variant System Prompt:
You are Qwen, an LLM to help generate multiple variants of a question.
Given a question and the context, rephrase it in 10 different ways without altering its meaning.
Output Format:
{
  "question": [new_question_1, new_question_2, ..., new_question_10]
}
```

### Context Retrieval
- An **Integrated Retriever** was employed to fetch relevant context for each query. This ensured that the generated responses were grounded in accurate and specific information.

### Response Generation
- A larger LLM was tasked with generating responses for each *(question, context)* pair.  
- This process followed the system prompt:

```plaintext
SFT Dataset Creator System Prompt:
You are Qwen, a SFT Dataset creation tool. Given a question and context related to the question, you generate responses based on them.
Output Format:
Question:
{question}
Context:
{context}
Response:
{generated_response}
```

### Topics and Course Coverage
- The dataset comprehensively covered topics such as **Machine Learning**, **Deep Learning**, **Data Science**, **Programming**, and **Development**.
- It included a broad spectrum of courses, such as:
  - Foundations of Artificial Intelligence
  - Algorithms
  - Deep Learning
  - Natural Language Processing
  - Web Development
  - Fundamentals of Cloud Computing
  - And more...

#### Dataset Composition
Each entry in the dataset includes:
- **Question:** A query or its variant.  
- **Context:** Information retrieved for the query.  
- **Response:** The generated answer to the query, ensuring alignment with the provided context.

#### Dataset Repository
The synthetic dataset has been versioned and published in the Hugging Face repository. It can be accessed at the following link: **Curriculum Compass SFT Dataset**

#### Conclusion (Dataset)
This synthetic dataset serves as a foundational resource for transferring knowledge from larger to smaller LLMs. By leveraging diverse queries, context retrieval, and robust response generation, the dataset enables fine-tuning smaller LLMs for enhanced domain-specific performance.


### Evaluation

For the Curriculum Compass project, we evaluated the RAG (Retrieval-Augmented Generation) system using a comprehensive set of metrics to assess its effectiveness and reliability. The evaluation employed both traditional n-gram-based metrics, such as ROUGE and BLEU, and semantic evaluation metrics like BERTScore. N-gram metrics like ROUGE and BLEU provided insight into the lexical overlap between the generated responses and the ground truth, measuring precision and recall at different granularity levels. While these metrics offer a straightforward and interpretable way to gauge performance, they may not fully capture the semantic coherence of the responses, particularly for nuanced or paraphrased answers. BERTScore addressed this limitation by leveraging contextual embeddings to evaluate the semantic similarity of the generated content to the reference answers, offering a more robust assessment of the RAG system's ability to produce contextually relevant responses.

Additionally, we used LLM-as-a-judge to evaluate the RAG system across the RAG triad metrics: relevance, accuracy, and groundedness. This approach involved leveraging large language models to holistically assess the quality of responses based on their alignment with retrieved knowledge, factual accuracy, and adherence to the source material. By incorporating synthetic datasets specifically designed for the task, we ensured that the evaluation accounted for diverse scenarios and query complexities. This dual-layer evaluation approach—combining traditional and modern metrics—provided a well-rounded view of the system’s performance, enabling us to identify areas of strength and opportunities for improvement in both retrieval and generation components.
