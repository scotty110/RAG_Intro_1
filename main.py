import mlx.core as mx
from mlx_lm import load as load_llm, generate
from mlx_embeddings import load as load_embedding
from qdrant_client import QdrantClient

# --- Configuration ---
QDRANT_PATH = "./data_prep/buffet_db"
COLLECTION_NAME = "buffet_rag_collection"
EMBEDDING_MODEL_NAME = "mlx-community/embeddinggemma-300m-8bit"
#LLM_MODEL_NAME = "mlx-community/gpt-oss-20b-MXFP4-Q8"
LLM_MODEL_NAME = "mlx-community/gemma-3n-E4B-it-lm-4bit"

# --- 1. Retrieval Functions ---

def embed_query(query: str, model, tokenizer):
    """Creates an embedding for a single query."""
    prefix = "task: search result | query: "
    encoded_input = tokenizer._tokenizer([prefix + query], return_tensors="mlx")
    output = model(encoded_input['input_ids'], encoded_input['attention_mask'])
    return output.text_embeds[0]

def retrieve_context(query_embedding, client: QdrantClient, top_k=3):
    """Searches Qdrant for the most relevant context."""
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )
    context = [hit.payload['text'] for hit in search_result]
    return "\n\n".join(context)

# --- 2. Augmentation & Generation ---

def build_prompt(query: str, context: str) -> str:
    """Builds the final prompt for the language model."""
    return (
        """
            You are a trusted **business partner and executive advisor** to the CEO.  
            Your role is to provide **both emotional and strategic support** so they can perform at their best.  
            When context is available, use it to ground your advice in the company’s real data, documents, and past decisions.  

            ## Core Principles
            1. **Dual Lens** – Balance **financial/business advice** with **emotional intelligence**. Recognize that CEOs make decisions under pressure and need both numbers-driven clarity and personal grounding.  
            2. **Strategic Depth** – Provide structured, forward-looking recommendations (financial planning, market strategy, organizational dynamics). Where context is available, cite specifics from company data to make advice actionable.  
            3. **Emotional Awareness** – Normalize the CEO’s challenges, offer perspective, and reinforce resilience. Tailor tone: calm in crisis, empowering when stakes are high, direct but empathetic when feedback is needed.  
            4. **Contextual Integration** – If company documents, financials, or reports are available, use them to:  
            - Highlight risks/opportunities.  
            - Compare decisions to benchmarks or past outcomes.  
            - Bring clarity to trade-offs.  
            5. **Confidentiality & Trust** – Always act as a discreet, loyal partner whose only goal is to help the CEO succeed.  

            ## Guidance Framework
            - **Understand the situation**: Probe gently if context is missing.  
            - **Address emotions first**: Acknowledge stress, excitement, or uncertainty before shifting to numbers or strategy.  
            - **Analyze facts**: Where context is present, ground insights in actual company metrics/documents. Otherwise, give general best practices.  
            - **Offer options**: Present at least two viable paths forward (e.g., conservative vs. aggressive strategy), with trade-offs.  
            - **Reinforce agency**: End by empowering the CEO, reminding them they are capable of making strong decisions.  .\n\n
        """
        "--- CONTEXT ---\n"
        f"{context}\n"
        "--- QUESTION ---\n"
        f"{query}"
    )

def answer_question(query: str, llm_model, llm_tokenizer, embed_model, embed_tokenizer, qdrant_client):
    """Orchestrates the entire RAG process with streaming generation."""
    print("Step 1: Creating query embedding...")
    query_embedding = embed_query(query, embed_model, embed_tokenizer)
    
    print("Step 2: Retrieving context from database...")
    retrieved_context = retrieve_context(query_embedding, qdrant_client)
    
    print("Step 3: Building prompt and generating answer...")
    prompt = build_prompt(query, retrieved_context)
    
    if llm_tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    print("\n--- FINAL ANSWER ---\n")
    out = []
    # Use the generate function as a generator to stream tokens
    for chunk in generate(llm_model, llm_tokenizer, prompt=prompt, verbose=False, max_tokens=4096):
        out.append(chunk)
        print(chunk, end="", flush=True)
    
    print() # Add a newline after the response
    final_answer = "".join(out)
    
    return final_answer, retrieved_context

# --- Main Execution ---

if __name__ == "__main__":
    print("Loading models... (This may take a moment)")
    llm_model, llm_tokenizer = load_llm(LLM_MODEL_NAME)
    embed_model, embed_tokenizer = load_embedding(EMBEDDING_MODEL_NAME)
    
    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    #user_query = "What did Warren Buffett say about long-term investing?"
    user_query = "How might a financial analyst creatively leverage unexpected quarterly budget discrepancies to innovate or enhance service offerings for a small business while ensuring long-term financial stability?"
    
    print(f"\nAnswering question: \"{user_query}\"\n" + "="*50)
    
    # The answer is now streamed to the console from within this function call
    final_answer, context = answer_question(
        user_query,
        llm_model,
        llm_tokenizer,
        embed_model,
        embed_tokenizer,
        qdrant_client
    )
    
    print("\n" + "="*50)
    print("\n--- RETRIEVED CONTEXT ---\n")
    print(context)
    print("\n" + "="*50)
