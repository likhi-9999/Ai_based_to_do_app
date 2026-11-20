import os
import json
import sqlite3
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Optional GenAI support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE = "todo.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="AI Smart To-Do App",
    description="Interview-ready GenAI To-Do application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            priority TEXT NOT NULL DEFAULT 'medium',
            completed INTEGER NOT NULL DEFAULT 0,
            due_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    priority: str = "medium"
    due_date: str | None = None


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    priority: str | None = None
    due_date: str | None = None
    completed: bool | None = None


class NaturalLanguageTask(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)


class AIRequest(BaseModel):
    task_id: int


# ============================================================
# HELPER FUNCTIONS
# ============================================================

VALID_PRIORITIES = {
    "low",
    "medium",
    "high",
    "urgent"
}


def validate_priority(priority):
    if priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=400,
            detail="Priority must be low, medium, high, or urgent."
        )


def row_to_dict(row):
    if row is None:
        return None

    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "priority": row["priority"],
        "completed": bool(row["completed"]),
        "due_date": row["due_date"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


def get_task_or_404(task_id):
    connection = get_connection()

    row = connection.execute(
        "SELECT * FROM tasks WHERE id = ?",
        (task_id,)
    ).fetchone()

    connection.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found."
        )

    return row


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Smart To-Do App is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# CREATE TASK
# ============================================================

@app.post("/tasks")
def create_task(task: TaskCreate):

    validate_priority(task.priority)

    now = datetime.now().isoformat()

    connection = get_connection()

    cursor = connection.execute("""
        INSERT INTO tasks
        (
            title,
            description,
            priority,
            completed,
            due_date,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        task.title,
        task.description,
        task.priority,
        0,
        task.due_date,
        now,
        now
    ))

    connection.commit()

    task_id = cursor.lastrowid

    row = connection.execute(
        "SELECT * FROM tasks WHERE id = ?",
        (task_id,)
    ).fetchone()

    connection.close()

    return row_to_dict(row)


# ============================================================
# GET ALL TASKS
# ============================================================

@app.get("/tasks")
def get_tasks(
    search: str | None = None,
    priority: str | None = None,
    completed: bool | None = None
):

    connection = get_connection()

    query = "SELECT * FROM tasks WHERE 1=1"
    parameters = []

    if search:
        query += " AND (title LIKE ? OR description LIKE ?)"
        search_value = f"%{search}%"
        parameters.extend([
            search_value,
            search_value
        ])

    if priority:
        validate_priority(priority)
        query += " AND priority = ?"
        parameters.append(priority)

    if completed is not None:
        query += " AND completed = ?"
        parameters.append(int(completed))

    query += """
        ORDER BY
        CASE priority
            WHEN 'urgent' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 4
        END,
        created_at DESC
    """

    rows = connection.execute(
        query,
        parameters
    ).fetchall()

    connection.close()

    return [row_to_dict(row) for row in rows]


# ============================================================
# GET ONE TASK
# ============================================================

@app.get("/tasks/{task_id}")
def get_task(task_id: int):

    row = get_task_or_404(task_id)

    return row_to_dict(row)


# ============================================================
# UPDATE TASK
# ============================================================

@app.put("/tasks/{task_id}")
def update_task(
    task_id: int,
    task: TaskUpdate
):

    get_task_or_404(task_id)

    values = task.model_dump(exclude_unset=True)

    if "priority" in values:
        validate_priority(values["priority"])

    if not values:
        raise HTTPException(
            status_code=400,
            detail="No fields provided for update."
        )

    connection = get_connection()

    set_parts = []
    parameters = []

    for field, value in values.items():

        if field == "completed":
            value = int(value)

        set_parts.append(f"{field} = ?")
        parameters.append(value)

    set_parts.append("updated_at = ?")
    parameters.append(datetime.now().isoformat())

    parameters.append(task_id)

    query = f"""
        UPDATE tasks
        SET {", ".join(set_parts)}
        WHERE id = ?
    """

    connection.execute(
        query,
        parameters
    )

    connection.commit()

    row = connection.execute(
        "SELECT * FROM tasks WHERE id = ?",
        (task_id,)
    ).fetchone()

    connection.close()

    return row_to_dict(row)


# ============================================================
# COMPLETE / UNCOMPLETE TASK
# ============================================================

@app.patch("/tasks/{task_id}/complete")
def toggle_complete(task_id: int):

    row = get_task_or_404(task_id)

    new_status = 0 if row["completed"] else 1

    connection = get_connection()

    connection.execute("""
        UPDATE tasks
        SET completed = ?, updated_at = ?
        WHERE id = ?
    """, (
        new_status,
        datetime.now().isoformat(),
        task_id
    ))

    connection.commit()

    row = connection.execute(
        "SELECT * FROM tasks WHERE id = ?",
        (task_id,)
    ).fetchone()

    connection.close()

    return row_to_dict(row)


# ============================================================
# DELETE ONE TASK
# ============================================================

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):

    get_task_or_404(task_id)

    connection = get_connection()

    connection.execute(
        "DELETE FROM tasks WHERE id = ?",
        (task_id,)
    )

    connection.commit()
    connection.close()

    return {
        "message": "Task deleted successfully."
    }


# ============================================================
# DELETE ALL TASKS
# ============================================================

@app.delete("/tasks")
def delete_all_tasks():

    connection = get_connection()

    cursor = connection.execute(
        "DELETE FROM tasks"
    )

    deleted_count = cursor.rowcount

    connection.commit()
    connection.close()

    return {
        "message": "All tasks deleted.",
        "deleted_count": deleted_count
    }


# ============================================================
# STATISTICS
# ============================================================

@app.get("/statistics")
def statistics():

    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) FROM tasks"
    ).fetchone()[0]

    completed = connection.execute(
        "SELECT COUNT(*) FROM tasks WHERE completed = 1"
    ).fetchone()[0]

    pending = connection.execute(
        "SELECT COUNT(*) FROM tasks WHERE completed = 0"
    ).fetchone()[0]

    urgent = connection.execute(
        "SELECT COUNT(*) FROM tasks WHERE priority = 'urgent'"
    ).fetchone()[0]

    high = connection.execute(
        "SELECT COUNT(*) FROM tasks WHERE priority = 'high'"
    ).fetchone()[0]

    connection.close()

    completion_rate = (
        round((completed / total) * 100, 2)
        if total > 0
        else 0
    )

    return {
        "total": total,
        "completed": completed,
        "pending": pending,
        "urgent": urgent,
        "high_priority": high,
        "completion_rate": completion_rate
    }


# ============================================================
# GENAI CLIENT
# ============================================================

def get_ai_client():

    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="OpenAI package is not installed."
        )

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Set the environment variable first."
            )
        )

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# AI: NATURAL LANGUAGE TASK CREATION
# ============================================================

@app.post("/ai/create-task")
def ai_create_task(request: NaturalLanguageTask):

    client = get_ai_client()

    prompt = f"""
Convert the following natural-language request into a task.

User request:
{request.text}

Return ONLY valid JSON in this exact structure:

{{
    "title": "short task title",
    "description": "useful description",
    "priority": "low|medium|high|urgent",
    "due_date": null
}}

If the user specifies a deadline, return an ISO date/time.
If there is no deadline, return null.
Do not invent information.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent "
                        "productivity assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = json.loads(
            response.choices[0].message.content
        )

        return {
            "success": True,
            "task": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {str(error)}"
        )


# ============================================================
# AI: TASK BREAKDOWN
# ============================================================

@app.post("/ai/breakdown")
def ai_breakdown(request: AIRequest):

    row = get_task_or_404(request.task_id)

    client = get_ai_client()

    prompt = f"""
Break the following task into practical subtasks.

Task title:
{row["title"]}

Description:
{row["description"] or "None"}

Return ONLY JSON:

{{
    "subtasks": [
        {{
            "title": "subtask",
            "description": "short explanation"
        }}
    ]
}}

Return between 3 and 7 actionable subtasks.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert "
                        "productivity assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = json.loads(
            response.choices[0].message.content
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AI breakdown failed: {str(error)}"
        )


# ============================================================
# AI: PRIORITIZE TASK
# ============================================================

@app.post("/ai/prioritize")
def ai_prioritize(request: AIRequest):

    row = get_task_or_404(request.task_id)

    client = get_ai_client()

    prompt = f"""
Analyze this task and recommend a priority.

Title:
{row["title"]}

Description:
{row["description"] or "None"}

Due date:
{row["due_date"] or "None"}

Return ONLY JSON:

{{
    "priority": "low|medium|high|urgent",
    "reason": "short explanation"
}}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI productivity "
                        "and prioritization assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = json.loads(
            response.choices[0].message.content
        )

        validate_priority(result["priority"])

        return result

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AI prioritization failed: {str(error)}"
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
