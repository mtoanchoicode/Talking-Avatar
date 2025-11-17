# 🤖 3D Avatar Communication System

This repository contains the code for a **real-time, 3D avatar communication system**, featuring a **React frontend** and a **Python backend**.

---

## 📁 Project Structure

The project uses a standard structure separating frontend assets (React application) and core backend logic (Python server).

| Directory/File           | Description                                                                                                  |
| :----------------------- | :----------------------------------------------------------------------------------------------------------- |
| `talking_avatar/`        | Root directory for the project.                                                                              |
| ├── `audio_output/`      | Output directory for generated audio files.                                                                  |
| ├── `node_modules/`      | Automatically generated directory containing Node.js dependencies. **(Ignored by Git)**                      |
| ├── `public/`            | Static assets directory for the React frontend (e.g., `index.html`).                                         |
| ├── `RAG_index/`         | Directory likely used for storing data or indices for Retrieval-Augmented Generation (RAG) functionality.    |
| ├── `src/`               | Source code for the React frontend.                                                                          |
| │   ├── `App.js`         | Main React component for the user interface.                                                                 |
| │   ├── `blendData.json` | Configuration for avatar blend shapes/animations.                                                            |
| │   ├── `index.js`       | Entry point for the React application.                                                                       |
| │   └── `...`            | Other frontend assets and components.                                                                        |
| ├── `.env`               | **Crucial configuration file.** Stores API keys and environment variables. **(Must be obtained separately)** |
| ├── `.gitignore`         | Specifies files and directories to ignore in version control (like `node_modules`).                          |
| ├── `package.json`       | Defines Node.js dependencies and scripts for the frontend.                                                   |
| ├── `requirements.txt`   | Defines Python dependencies for the backend.                                                                 |
| ├── `server.py`          | Main Python backend application script.                                                                      |

---

## 🚀 Setup and Run Instructions

This project requires both **Node.js/npm** for the frontend and **Python/pip** for the backend.

### 1. Configuration (Crucial Step)

The application requires environment variables (e.g., API keys) to function.

> **Download the `.env` file:**
>
> - This file is **not included** in the repository for security reasons.
> - Contact the project owner/administrator to obtain the necessary `.env` file.
> - Place the obtained `.env` file directly in the root directory (`talking_avatar/`).

### 2. Install Dependencies

You must install dependencies for both the Node.js frontend and the Python backend.

#### A. Frontend Dependencies (Node.js)

Run this command in the root directory (`talking_avatar/`):

```bash
npm install
```
