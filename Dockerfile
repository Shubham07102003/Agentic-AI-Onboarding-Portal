FROM node:20-bullseye as frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* frontend/pnpm-lock.yaml* ./
RUN npm ci || true && npm install
COPY frontend/ .
RUN npm run build

FROM python:3.10-slim as backend
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
COPY --from=frontend /frontend/dist /app/static
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
