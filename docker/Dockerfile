FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependencies
COPY pyproject.toml /app/
COPY .python-version /app/
COPY uv.lock /app/

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

# Copy the application into the container.
COPY . /app

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]
