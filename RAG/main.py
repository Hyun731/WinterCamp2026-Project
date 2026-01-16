import os
from google import genai
from fastapi import FastAPI
import chromadb
from chromadb.utils import embedding_functions

embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cuda"
)

client = chromadb.PersistentClient(path="./DB")
precedent_col = client.get_collection(name="precedents", embedding_function=embedding_model)
law_col = client.get_collection(name="laws", embedding_function=embedding_model)

app = FastAPI()

api_key = os.environ.get("API")
if not api_key:
    raise ValueError("API 환경 변수가 설정되지 않았습니다.")

client_gemini = genai.Client(api_key=api_key)
model_id = "gemini-2.5-flash"

@app.post("/answer")
def answer(query: str, results_num: int = 4):
    law_results = law_col.query(query_texts=query, n_results=results_num)
    law_context = "\n".join(law_results['documents'][0])

    results = precedent_col.query(query_texts=query, n_results=results_num)
    context = "\n".join(results['documents'][0])

    combined_context = f"관련 법령\n{law_context}\n\n참고 판례\n{context}"

    prompt = f'''System: You are a professional legal expert in South Korean law. 
    Analyze the provided [Legal Information] and provide a comprehensive response to the user's question.
    
    [Instructions]:
    1. Output Language: Korean. All explanations must be written in clear and professional Korean.
    2. Base your answer strictly on the provided [Legal Information]. If the information is insufficient, state that "관련 근거를 찾을 수 없습니다."
    3. Cite specific Statutes and Articles (e.g., "민법 제544조").
    4. Include details from Court Precedents (e.g., "대법원 판결" or "고등법원 판례") to support your reasoning.
    5. Structure your response logically: 
       - Summary of the legal situation
       - Applicable laws and analysis of precedents
       - Final conclusion and advice
    6. Explain technical legal terms in a way that is easy for a layperson to understand.
    
    [Legal Information (Context)]:
    {combined_context}
    
    [User Question]: {query}
    
    Final Response (in Korean):'''

    try:
        response = client_gemini.models.generate_content(
            model=model_id,
            contents=prompt
        )

        return {
            "answer": response.text,
            "source_laws": law_results['metadatas'][0],
            "source_precedents": results['metadatas'][0]
        }
    except Exception as e:
        return {"error": str(e)}