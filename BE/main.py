import os

import pymupdf4llm
import requests

from fastapi import FastAPI, Body, UploadFile, File
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response, RedirectResponse, JSONResponse
from fastapi import HTTPException
from config import GEMINI_API_KEY as gemini_api_key

# from backend import rag_server
# from backend.rag_server import llm_response

app = FastAPI()

# RAG 서버 URL
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag_project:7777")

#동언이 gemini_key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", gemini_api_key)
full_text = ""

# 1번 서버 띄우기
@app.get("/")
def hello():
    return {"message": "Hello World"}

#3. 로그인 하기 위해서 클래스 생성    #basemodel -> type 강제
class LoginUser(BaseModel):
    username: str
    password: str

users = []
users.append(LoginUser(username="park",password="q1w2e3"))
users.append(LoginUser(username="choi",password="q1w2e3"))
users.append(LoginUser(username="admin",password="12345678"))
historys = {}
historys["park"] = []
historys["choi"] = []
historys["admin"] = []
#2번 로그인
@app.post("/login")
def login(response: Response, user: LoginUser = Body()): #option enter  response ->sh,  body ->
    # 로그인 검증
    ok = any(u.username == user.username and u.password == user.password for u in users)
    if not ok:
        return JSONResponse({"ok": False, "reason": "invalid credentials"}, status_code=401)

    # 응답 만들고 쿠키 세팅
    res = JSONResponse({"ok": True})
    res.set_cookie("username", user.username, httponly=True)
    return res


@app.get("/page")
def page(request: Request):
    username = request.cookies.get("username")  # KeyError 방지
    if not username:
        return JSONResponse({"ok": False, "reason": "no cookie"}, status_code=401)

    # username이 등록된 유저인지 확인
    if username in [u.username for u in users]:
        return {"ok": True, "message": f"welcome {username}"}

    return JSONResponse({"ok": False, "reason": "unknown user"}, status_code=403)


def get_current_user(request: Request) -> str:
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")

    if username not in [u.username for u in users]:
        raise HTTPException(status_code=401, detail="다시 로그인해주세요")

    return username

def upload_to_rag(full_text: str):
    response = requests.post(
        f"{RAG_SERVER_URL}/upload",
        json={"full_text": full_text},

        timeout=60
    )
    response.raise_for_status()
    return response.json()

def gemini_response(question: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {"parts": [{"text": question}]}
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    return data["candidates"][0]["content"]["parts"][0]["text"]


def llm_response(question: str):
    response = requests.post(
        f"{RAG_SERVER_URL}/answer",
        json={"query": question},
        timeout=180
    )
    response.raise_for_status()
    return response.json()

@app.post("/query")
def query(request: Request, query : str = Body(...,embed=True)):
    # 1) 쿠키 사용자 인증
    username = get_current_user(request)
    # RAG server에 텍스트 업로
    # RAG server에서 refer 검색 및 LLM 응답 생성

    lr = llm_response(query)
    # lr = gemini_response(query)
    historys[username].append({query: lr})
    return {
        "ok": True,
        "user": username,
        "query": query,
        "llm_response": lr
    }

@app.get("/history")
def history(request: Request):
    username = get_current_user(request)
    user_query = historys[username]
    user_query = user_query[-10:]
    result = ""
    for number,item in enumerate(user_query,start=1):
        question = ""
        answer = ""
        for x in item:
            question = x
            answer = item[x]
        result += f'{number}: query : {question}\nanswer : {answer}\n '

    return result

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)