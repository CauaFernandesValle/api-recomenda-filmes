#importa o pandas (leitor de cvs) e da o apelido de pd pra ele
import pandas as pd

#Importa a ferramenta TfidfVectorizer do scikit-learn.
#Ela é usada para converter texto (sinopses) em uma matriz de números.
from sklearn.feature_extraction.text import TfidfVectorizer

#importa a ferramente cosine_similarity que calcula a similaridade entre os cossenos para calcular a similaridade de dois vetores
from sklearn.metrics.pairwise import cosine_similarity

#cria as variaveis
df = None
cosine_sim = None
indices = None

def load_data_and_build_model():

    global df, cosine_sim, indices

    try:
        #cria a variavel df (dataframe) que recebe a funçao que le e armazena todo o csb=v
        df = pd.read_csv("movies_metadata.csv", low_memory=False)
        df = df.sample(5000, random_state=42)
        df['overview'] = df['overview'].fillna('').astype(str)
    except FileNotFoundError:
        print("ERRO: O arquivo 'movies_metadata.csv' não foi encontrado.")
        print("Por favor, baixe o dataset e coloque-o na mesma pasta.")
        return

    #função que ignora toda linha do df que tenha os campos title ou overview nulos
    df.dropna(subset=['title','overview'], inplace=True)

    #cria uma instancia do tfidf vetorizador e pede pra ignorar as palavras comuns do ingles como the, in
    tfidf = TfidfVectorizer(stop_words="english")

    #essa funçao pega os dado de overview do df e faz todos os calculos para numerar as palavras e tal
    #e armazena na variavel tfidf_matrix
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    #calcula a similaridade de toda linha de tfidf_matrix com outra, e armazena na matriz cosine_sim
    #a matriz cosine_sim armazena cada linha sendo uma linha de tfidf_matrix e cada coluna de cosine_sim
    #os mesmo filmes, de forma que a interseção entre linha e coluna mostre o quanto perto do cosseno cada
    #linha x coluna esta. exemplo, a linha/coluna 1 armazena o filme 1, e a linha/coluna 5 armazena o filme 5
    #a pocisao 1,5 e 5,1 da matriz armazena a similaridade desses dois filmes
    cosine_sim = cosine_similarity(tfidf_matrix)

    df = df.reset_index()
    #da um valor de index para o titulo de cada filme
    indices = pd.Series(df.index,df['title']).drop_duplicates()

    print("Modelo de recomendação carregado com sucesso.")

#função que retorna as recomendações
def get_recommendations(titulo_base):

    if cosine_sim is None or indices is None:
        return {"error": "O modelo ainda não foi carregado."}

    try:
        #busca o indice do titulo base
        indice_base = indices[titulo_base]
    except KeyError:
        # Se o filme não está no nosso "mapa"
        return {"error": f"Filme '{titulo_base}' não encontrado no banco de dados."}
    
    if isinstance(indice_base, pd.Series):
        indice_base = indice_base.iloc[0]

    #cria uma lista onde cada elemento é um par do indece e do valor dele ex: [(0, 0.7), (1, 0.8), ...]
    lista_semelhante = list(enumerate(cosine_sim[indice_base]))

    #sorted(lista que sera ordenada, qual parte é pra ser considerada na ordenaçao 
    # (ja que essa lista e composta por dos numeros em cada indice), direção da ordenação)
    lista_ordenada = sorted(lista_semelhante, key=lambda x: x[1], reverse=True)

    #cria uma lista começando do indice 1 (pula o 0) da lista ordenada e indo até o indice 10
    lista_final = lista_ordenada[1:11]

    #cria uma lista só com o elemento indice de cada elemento de lista_final
    lista_de_indices = [ indice for indice, score in lista_final ]

    #vai apenas na coluna 'title' e pega os elementos de indice armazenados na lista_de_indices
    lista_titulos = df['title'].iloc[lista_de_indices]

    return {"recommendations": list(lista_titulos)}
