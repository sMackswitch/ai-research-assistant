import chromadb
from google import genai

# 1. Connect to our local database folder and the Gemini API
client = genai.Client()
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_collection(name="research_docs")

def ask_assistant(user_question):
    print(f"\nSearching database for: '{user_question}'...")
    
    # 2. Turn your question into a vector embedding
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=user_question
    )
    question_vector = response.embeddings[0].values
    
    # 3. Query ChromaDB to find the 3 most contextually relevant chunks
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3
    )
    
    # Extract the raw text from those matching chunks
    retrieved_chunks = results['documents'][0]
    
    # Glue the snippets together into a single block of context
    context = "\n---\n".join(retrieved_chunks)
    
    # 4. Craft the prompt telling Gemini to strictly use the book data
    system_prompt = (
        "You are an AI Research Assistant. Use the provided context snippets to answer the user's question. "
        "If the answer cannot be found in the context, say 'I cannot find that information in the uploaded documents.' "
        "Do not make things up. Break down complex physics concepts into clear, understandable terms."
    )
    
    full_prompt = f"Context:\n{context}\n\nQuestion: {user_question}"
    
    print("Synthesizing answer with Gemini...")
    
    # 5. Generate the final answer using gemini-2.5-flash
    ai_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_prompt,
        config={'system_instruction': system_prompt}
    )
    
    return ai_response.text

if __name__ == "__main__":
    # 💡 CHANGE THIS QUERY TO ANYTHING YOU WANT TO ASK YOUR QUANTUM BOOK!
    test_query = "What is the Schrodinger equation and what does it describe?"
    
    answer = ask_assistant(test_query)
    print("\n=== ASSISTANT ANSWER ===")
    print(answer)