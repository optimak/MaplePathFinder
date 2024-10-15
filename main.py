from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

app = FastAPI()

class UserInput(BaseModel):
    age: int
    education: str
    work_experience: int
    language_score: int

# Define your LangChain-based QA function
@app.post("/api/check-eligibility")
async def check_eligibility(input: UserInput):
    # Create user query dynamically
    user_query = f"I am {input.age} years old with a {input.education} degree, {input.work_experience} years of work experience, and a language score of {input.language_score}. What Canadian immigration pathways am I eligible for?"

    # Load the QA chain (using OpenAI and LangChain)
    llm = OpenAI(temperature=0)  # You can add your API key in environment variables or as an argument
    chain = load_qa_chain(llm)

    # Pass the query to LangChain to fetch results
    result = chain.run(question=user_query)

    return {"eligibility": result}
