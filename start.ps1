Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Agentic Multimodal GenAI Chatbot'; uv run backend\main.py"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Agentic Multimodal GenAI Chatbot\frontend'; npm run dev"
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"
