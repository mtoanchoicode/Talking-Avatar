# 🤖 3D Avatar Communication System

This repository contains the code for a **real-time, 3D avatar communication system**, featuring a **React frontend** and a **Python backend**.

---

## 📁 Project Structure

The project uses a structured layout separating the React frontend and the Python backend.

| Directory/File             | Description                                                                            |
| :------------------------- | :------------------------------------------------------------------------------------- |
| `talking_avatar/`          | Root directory for the project.                                                        |
| ├── `audio_output/`        | Output directory for generated audio files.                                            |
| ├── `node_modules/`        | Automatically generated Node.js dependencies. **(Ignored by Git)**                     |
| ├── `public/`              | Static assets directory for the React frontend.                                        |
| ├── `RAG_index/`           | Likely used for storing RAG-related data or indices.                                   |
| ├── `src/`                 | Source code for the React frontend.                                                    |
| │ ├── `App.js`             | Main React interface component.                                                        |
| │ ├── `blendData.json`     | Avatar blendshape/animation configuration.                                             |
| │ ├── `index.js`           | Entry point of the React application.                                                  |
| │ └── `...`                | Other frontend components and assets.                                                  |
| ├── `backend/`             | **Backend directory containing the Python server files.**                              |
| │ ├── `delete_pinecone.py` | Script related to delete all record in Pinecone.                                       |
| │ ├── `requirements.txt`   | Python dependencies for the backend.                                                   |
| │ └── `server.py`          | Main backend server application.                                                       |
| ├── `.env`                 | Critical environment variables (API keys, config). **Not included—obtain separately.** |
| ├── `.gitignore`           | Git ignore rules (e.g., `node_modules/`).                                              |
| ├── `package.json`         | Node.js dependencies and scripts for the frontend.                                     |

---

## 🚀 Setup and Run Instructions

This project requires **Node.js/npm** for the frontend and **Python/pip** for the backend.

---

## 1. Configuration (Important)

The project requires a `.env` file with environment variables.

> **Get the `.env` file from the project owner.**  
> Place it in the root directory:

### 2. Install Dependencies

You must install dependencies for both the Node.js frontend and the Python backend.

#### A. Frontend Dependencies (Node.js)

Run this command in the root directory (`talking_avatar/`):

```bash
npm install
```

#### B. Backend Dependencies (Node.js)

Run this command in the backend directory (`talking_avatar/`):

```bash
pip install -r requirements.txt
```

### 3. Run Application

#### A. Frontend (Node.js)

Run this command in the root directory (`talking_avatar/`):

```bash
npm start
```

#### B. Backend (Node.js)

Run this command in the backend directory (`talking_avatar/`):

```bash
python server.py
```
