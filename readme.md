### Execute app
```commandline
uvicorn api.py:app --host 0.0.0.0 --port 8000
```

# 1. Build the image
```commandline
podman build -t charaka-ai .
```

# 2. Run the container (maps port 8000)
```commandline
podman run -d -p 8000:8000 --name charaka-ai-container charaka-ai
```

# 3. View logs
```commandline
podman logs -f charaka-ai-container
```

# 4. Stop when done
```commandline
podman stop charaka-ai-container
```

# 5. Remove container (optional)
```commandline
podman rm charaka-ai-container
```
