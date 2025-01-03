# import torch
import weave

# from curriculum_compass.naive_rag.reranker import Reranker
# from curriculum_compass.naive_rag.utils import load_model_and_tokenizer, initialize_chromadb_client, load_embedding_model

# Step 2: Load Model and Tokenizer
# model_name = "Qwen/Qwen2.5-3B-Instruct"
# model, tokenizer = load_model_and_tokenizer(model_name)

# Step 3: Initialize ChromaDB Client
# collection = initialize_chromadb_client("./chromadb").get_or_create_collection("naive_rag_embeddings")

# # embedding_model_name = 'all-MiniLM-L6-v2'
# embedding_model = load_embedding_model(embedding_model_name)

# reranker = Reranker(device="cuda" if torch.cuda.is_available() else "cpu")

# weave.init(project_name="Naive_RAG_Reviews")


# Step 3: RAG Pipeline
class ReviewsRAGPipeline:
    # SYSTEM_INSTRUCTION = """
    # You are Curriculum Compass, a chatbot dedicated to assisting Northeastern University graduate students with course registration for the spring 2025 semester.
    # You have access to the latest information on available graduate courses for the spring 2025 semester, faculty profiles, and summarized student feedback from previous semesters.
 
    # Your goals are:
    # 1. To provide accurate, up-to-date information without speculating. If you lack information about a course or question, clearly communicate that to the student.
    # 2. To maintain a positive, professional tone. If past student feedback includes criticism, you should still respond diplomatically, focusing on constructive or neutral aspects.
    # 3. To be concise and relevant in your responses, helping students make informed decisions about their course choices.

    # Important Guidelines to be followed:
    # 1. The context is provided to you after retrieving reviews similar to the query being asked using a RAG pipeline.
    # Sometimes, the context is not relevant to the particular query being asked. You should always check if the context is related to the query, else reply that you don't have enough information to reply.
    # For example, the query could be about a particular course or professor, while the context would be of some other courses or professors, you should reply that you don't have enough information to these cases.
    # 2. Avoid negative or speculative responses, and prioritize factual information over assumption.
    
    # Things to check:
    # 1. Students may not mention the names of the courses properly. Their input could have typo's, mistakes. For example, students could input 'PDP' instead of 
    # 'Programming Design Paradigm' or they could mention 'Raj Venkat' instead of the full name of the professor 'Rajagopal Venkatesaramani'

    # Answer the questions comprehensively using the reviews from the context by summarizing them to help the student.
    # """
    
    def __init__(self, embedding_model, collection,reranker):
        self.embedding_model = embedding_model
        self.collection = collection
        # self.model = model
        # self.tokenizer = tokenizer
        self.reranker = reranker

    @weave.op(name="retrieve_reviews")
    def retrieve(self, query, top_k=5):
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return results["documents"]

    @weave.op(name="rerank_reviews")
    def rerank(self, query, retrieved_docs, top_k):
        # Flatten the list of retrieved documents
        flattened_docs = [doc for sublist in retrieved_docs for doc in sublist]
        
        # Rerank using the reranker
        reranked_docs = self.reranker.rerank(query, flattened_docs, top_k=top_k)
        return reranked_docs

    # @weave.op
    # def generate_response(self, query, retrieved_docs):
    #     # Join reranked documents into a context string
    #     context = "\n".join(retrieved_docs)
        
    #     # Prepare messages
    #     messages = [
    #         {"role": "system", "content": self.SYSTEM_INSTRUCTION},
    #         {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"}
    #     ]
        
    #     # Tokenize and generate
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    #     model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    #     generated_ids = self.model.generate(
    #         **model_inputs,
    #         max_new_tokens=4098,
    #         temperature=0.1
    #     )
    #     # Remove input tokens from output to isolate generated text
    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
        
    #     response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #     return response

    @weave.op(name="process_review_query")
    def __call__(self, query: str, initial_k: int = 10, final_k: int = 5):
        print("Retrieving")
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query, initial_k)

        print("Reranking")
        # Step 2: Rerank the retrieved documents
        reranked_docs = self.rerank(query, retrieved_docs, final_k)

        # print("Generating Response")
        # # Step 3: Generate a response
        # return self.generate_response(query, reranked_docs)
        return reranked_docs


# # Step 4: Use the RAG Pipeline
# rag_pipeline = NaiveReviewsRAGPipeline(embedding_model, collection, model, tokenizer, reranker)

# with weave.attributes({'user_id': 's-kishore', 'env': 'testing'}):
#     # Example Query
#     query = "How difficult is Algorithms under Prof. Raj Venkat?"
#     response = rag_pipeline(query, top_k=5)
#     print(response)