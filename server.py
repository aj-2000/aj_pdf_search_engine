from fastapi import FastAPI, BackgroundTasks, HTTPException
import uuid
import pickle
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

from models import IndexBuildTask, SearchTask
from main import IndexBuilder, SearchEngine

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


def save_index(index_file, data):
    """Save index data to a file."""
    with open(index_file, 'wb') as f:
        pickle.dump(data, f)


def load_index(index_file):
    """Load index data from a file."""
    with open(index_file, 'rb') as f:
        return pickle.load(f)


class IndexTask:
    def __init__(self):
        self._cancel = False

    async def run(self, task: IndexBuildTask):
        try:
            if os.path.exists(task.index_file) and not task.update_index:
                pass
            else:
                print("Building new index...")
                indexer = IndexBuilder(task.mode)
                candidate_labels = ['label1', 'label2',
                                    'label3']  # Define your labels here
                index_data = indexer.build(task.docs_path, use_transformer=False,
                                           use_zero_shot=True, candidate_labels=candidate_labels)
                save_index(task.index_file, index_data)
                print(f"Index saved to {task.index_file}")
        finally:
            self._cancel = False

    def cancel(self):
        self._cancel = True


# Dictionary to keep track of tasks
tasks = {}


@app.post("/build-index")
async def build_index(taskConfig: IndexBuildTask, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task = IndexTask()
    tasks[task_id] = {"task": task, "status": "in_progress"}
    background_tasks.add_task(task.run, taskConfig)
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


def search_index(query: str, index_file: str, mode: str):
    index_data = load_index(index_file)
    search_engine = SearchEngine(index_data, mode)
    return search_engine.query(query)


@app.post("/query", response_model=List[SearchResults])
def query_index(task: SearchTask):
    try:
        results = search_index(task.query, task.index_file, task.mode)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
