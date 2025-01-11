from fastapi import FastAPI

from services.track.routes.infer import infer_router

app = FastAPI()

app.include_router(infer_router)


@app.get("/")
async def root():
    return {"message": "YOLO inference service is running"}
