API de Recomendação de Filmes (FastAPI + Scikit-learn)

Este é um projeto de portfólio que implementa um sistema de recomendação de filmes (baseado em conteúdo) e o serve através de uma API RESTful construída com FastAPI.

Funcionalidades
API RESTful: Criada com FastAPI.IA (ML): Usa `scikit-learn` (TfidfVectorizer e Cosine Similarity) para encontrar filmes similares com base em suas sinopses.
Manipulação de Dados: Usa `pandas` para carregar e limpar os dados.

Como Rodar o Projeto

Clone este repositório:
    ```bash
    git clone https://github.com/CauaFernandesValle/api-recomenda-filmes
    ```

Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

Baixe o Dataset:
    Este projeto requer o dataset `movies_metadata.csv` do Kaggle.
    Link para Download: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
    Coloque o arquivo `movies_metadata.csv` na raiz do projeto.

Execute a API:
    ```bash
    uvicorn main:app --reload
    ```

Acesse a documentação interativa:
    Abra seu navegador e acesse: `http://127.0.0.1:8000/docs`