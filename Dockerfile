FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip && pip install -e .
CMD ["python", "-c", "import gamache as m; print(m.hello())"]
