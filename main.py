from fastapi import FastAPI
from recommender import load_data_and_build_model, get_recommendations

app = FastAPI()

@app.on_event("startup")
def on_startup():
    load_data_and_build_model()
    print("Modelo carregado com sucesso!")

@app.get("/")
def root():
    return {"message": "Bem-vindo à API de Recomendação!"}

@app.get("/recommend/{movie_title}")
def recommend_movie(movie_title: str):
    return get_recommendations(movie_title)