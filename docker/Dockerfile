# Base image
FROM python:3.10-slim

# 1. Disable Python's output buffering. Useful for debugging.
# 2. Set the environment variables
# 3. LC_ALL and LANG: ensures that the application can handle Unicode characters correctly.
# 4. DEBIAN_FRONTEND: prevents interactive prompts during package installations.
# 5. BUILD_POETRY_LOCK: specify the location of a Poetry lock file.
ENV \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="${HOME}/venv" \
    PATH="${HOME}/venv/bin:${PATH}" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    BUILD_POETRY_LOCK="${HOME}/poetry.lock.build"

# Set up Poetry for dependency management.
RUN HOME=/tmp pip install --no-cache-dir poetry==1.7.1


# 1. Copy the `pyproject.toml` and lock files (e.g. `*.lock`) to the `/app` directory.
# 2. Set the current working directory to `/app`.
COPY ./pyproject.toml ./*.lock /app/
WORKDIR /app

# 1. Create a virtual environment,
# 2. Upgrade the pip package manager
# 3. Install project dependencies with poetry excluding the root package and dev dependencies.
# 4. Copy the dependency lock file
# 5. Clean up the cache directory.
RUN python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip \
    && poetry install --no-root --no-dev \
    && poetry run pip install torch==2.2.2 lightning==2.2.2 \
    && cp poetry.lock "${BUILD_POETRY_LOCK}" \
    && rm -rf "${HOME}/.cache/*"

# 1. Copy the Docker-related shell scripts to the root directory of the container.
# 2. Using root privileges, set the appropriate permissions for the scripts.
USER root
COPY ./docker/**/*.sh /
RUN chmod +x /*.sh

# 1. Switch to the directory where the project files are located.
COPY . /app/
# Entrypoint to start the MLflow server
CMD ["/startup-script.sh"]
