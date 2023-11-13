from fastapi import FastAPI, BackgroundTasks, HTTPException
import uuid
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PDFFile(BaseModel):
    name: str
    path: str


@app.get("/list-pdfs", response_model=List[PDFFile])
def list_pdfs(docs_path: str = "docs"):
    pdf_files = []
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(PDFFile(name=file, path=full_path))
    return pdf_files

# Mock function to represent index building


class IndexTask:
    def __init__(self):
        self._cancel = False

    async def run(self, docs_path: str, index_file: str):
        try:
            # Mock index building process
            for i in range(10):
                if self._cancel:
                    print(f"Index building cancelled for {docs_path}.")
                    return
                await asyncio.sleep(1)  # Simulating work
            print(
                f"Index built for documents in {docs_path} and saved to {index_file}")
        finally:
            self._cancel = False

    def cancel(self):
        self._cancel = True


# Dictionary to keep track of tasks
tasks = {}


@app.post("/build_index")
async def build_index(background_tasks: BackgroundTasks, docs_path: str = "docs", index_file: str = "index_data.pkl"):
    task_id = str(uuid.uuid4())
    task = IndexTask()
    tasks[task_id] = {"task": task, "status": "in_progress"}
    background_tasks.add_task(task.run, docs_path, index_file)
    return {"task_id": task_id}


@app.get("/index_status/{task_id}")
def get_index_status(task_id: str):
    task_info = tasks.get(task_id)
    if task_info:
        return {"status": task_info["status"]}
    raise HTTPException(status_code=404, detail="Task not found")


@app.post("/stop_index/{task_id}")
def stop_index(task_id: str):
    task_info = tasks.get(task_id)
    if task_info and task_info["status"] == "in_progress":
        task_info["task"].cancel()
        task_info["status"] = "cancelled"
        return {"status": "cancelled"}
    raise HTTPException(
        status_code=404, detail="Task not found or not in progress")


class SearchResults(BaseModel):
    path: str
    page: int
    score: float
    label: Optional[str] = None

# Assuming this function is adapted from your original script


def search_index(query: str, index_file: str) -> List[SearchResults]:
    # Implement the logic to search the index and return results
    # This function should return a list of SearchResults
    pass


@app.post("/query", response_model=List[SearchResults])
def query_index(query: str, index_file: str = "index_data.pkl"):
    try:
        results = search_index(query, index_file)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
