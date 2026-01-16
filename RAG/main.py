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
model_id = "gemini-3-flash"

@app.post("/answer")
def answer(query: str, results_num: int = 15):
    try:
        extract_prompt = f"다음 질문에서 법률 검색을 위한 핵심 키워드(단어)만 3~4개 뽑아줘: '{query}'"
        keywords_resp = client_gemini.models.generate_content(
            model=model_id,
            contents=extract_prompt
        )
        refined_query = keywords_resp.text.strip()

        law_results = law_col.query(query_texts=refined_query, n_results=results_num)
        law_context = "\n".join(law_results['documents'][0])

        precedent_results = precedent_col.query(query_texts=refined_query, n_results=results_num)
        context = "\n".join(precedent_results['documents'][0])

        combined_context = f"관련 법령\n{law_context}\n\n참고 판례\n{context}"

        prompt = f'''System: You are a professional legal expert in South Korean law. 
        Analyze the provided [Legal Information] and provide a comprehensive response to the user's question.
        
        [Instructions]:
        1. Output Language: Korean. All explanations must be written in clear and professional Korean.
        2. Base your answer on the provided [Legal Information]. 만약 정보가 다소 부족하더라도 법령의 일반 원칙에 비추어 논리적으로 설명하세요. 정말 관련이 없는 경우에만 "관련 근거를 찾을 수 없습니다"라고 답하세요.
        3. Cite specific Statutes and Articles (e.g., "민법 제544조").
        4. Include details from Court Precedents to support your reasoning.
        5. Structure: 상황 요약 -> 관련 법령/판례 분석 -> 최종 결론 및 조언.
        
        [Legal Information (Context)]:
        {combined_context}
        
        [User Question]: {query}
        
        Final Response (in Korean):'''

        response = client_gemini.models.generate_content(
            model=model_id,
            contents=prompt
        )

        return {
            "answer": response.text,
            "source_laws": law_results['metadatas'][0],
            "source_precedents": precedent_results['metadatas'][0],
            "refined_query": refined_query
        }

    except Exception as e:
        return {"error": str(e)}