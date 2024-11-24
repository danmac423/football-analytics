from fastapi import FastAPI
from services.track.routes.infer import infer_router
import uvicorn


app = FastAPI()

app.include_router(infer_router)


@app.get("/")
async def root():
    return {"message": "YOLO inference service is running"}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=5000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
