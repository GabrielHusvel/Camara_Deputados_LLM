
import matplotlib.pyplot as plt
import json
import os
import requests
import pandas as pd
import google.generativeai as genai
import requests
from datetime import datetime


def coletar_dados_deputados():
    """Coleta informações dos deputados atuais e salva como um arquivo Parquet."""
    url_base = "https://dadosabertos.camara.leg.br/api/v2/deputados"
    response = requests.get(url_base, params={'ordem': 'ASC', 'ordenarPor': 'id'})
    
    if response.status_code == 200:
        data = response.json()
        deputados_df = pd.DataFrame(data['dados'])
        os.makedirs('data', exist_ok=True)
        deputados_df.to_csv('data/deputados.csv', index=False)
        print("Dados dos deputados salvos em 'data/deputados.parquet'.")
    else:
        raise Exception(f"Erro ao acessar API: {response.status_code} - {response.text}")

# if __name__ == "__main__":
#     coletar_dados_deputados()


def gerar_grafico_pizza_partidos(arquivo_deputados, arquivo_saida):
    """
    Gera um gráfico de pizza mostrando a distribuição de deputados por partido.

    Args:
        arquivo_deputados (str): Caminho para o arquivo CSV contendo os dados dos deputados.
        arquivo_saida (str): Caminho para salvar o gráfico de pizza.
    """
    try:
        deputados_df = pd.read_parquet(arquivo_deputados) # Originalmente o LLM escreveu "pd.read_csv(arquivo_deputados, encoding='utf8')"
    except FileNotFoundError:
        print(f"Erro: Arquivo {arquivo_deputados} não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    contagem_partidos = deputados_df['siglaPartido'].value_counts()

    # Cria o gráfico de pizza
    plt.figure(figsize=(10, 10))  # Ajusta o tamanho da figura para melhor visualização
    plt.pie(contagem_partidos, labels=contagem_partidos.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuição de Deputados por Partido')


    # Adiciona os total e percentual como legenda fora do gráfico.
    total_deputados = len(deputados_df)
    legenda = [f'{partido}: {contagem} ({contagem/total_deputados*100:.1f}%)' for partido, contagem in contagem_partidos.items()]
    plt.legend(legenda, bbox_to_anchor=(1, 0.5), loc="center left") # Posiciona a legenda à direita


    plt.tight_layout() # Ajusta o layout para evitar que a legenda corte fora.
    plt.savefig(arquivo_saida)
    print(f"Gráfico salvo em {arquivo_saida}")
  

# # # Exemplo de uso (substitua pelo caminho real do seu arquivo)
# arquivo_deputados = 'data/deputados.parquet'
# arquivo_saida = 'data/distribuicao_deputados.png'
# gerar_grafico_pizza_partidos(arquivo_deputados, arquivo_saida)



# deputados = r"""
# PL: 93 (18.1%)
# PT: 68 (13.3%)
# UNIÃO: 59 (11.5%)
# PP: 50 (9.7%)
# REPUBLICANOS: 44 (8.6%
# PSD: 44 (8.6%)
# MDB: 44 (8.6%)
# PDT: 18 (3.5%)
# PSB: 14 (2.7%)
# PODE: 14 (2.7%)
# PSOL: 13 (2.5%)
# PSDB: 12 (2.3%)
# PCdoB: 7 (1.4%)
# AVANTE: 7 (1.4%)
# PV: 5 (1.0%)
# CIDADANIA: 5 (1.0%)
# SOLIDARIEDADE: 5 (1.0%
# PRD: 5 (1.0%)
# NOVO: 4 (0.8%)
# REDE: 1 (0.2%)
# S.PART.: 1 (0.2%)
# """
# prompt = (
#     '''
#     Você é um analista político experiente especialista em análise de dados. Use como base os dados {deputados} que contém a distribuição de deputados por partido na Câmara. Sua tarefa é gerar insights relevantes sobre como essa distribuição influencia as decisões e dinâmicas políticas na Câmara dos Deputados.

# - Analise as proporções de partidos representados.
# - Identifique quais partidos têm maior influência.
# - Considere a concentração de partidos e como isso pode impactar votações.
# - Sua resposta devera ser um JSON com os seguintes campos:
#   - "partidos": Lista dos partidos com maior representatividade e suas porcentagens.
#   - "concentração": Observações sobre a concentração dos partidos.
#   - "implicacoes": Como essa distribuição pode afetar as coalizões, votações e o andamento de projetos de lei.

# **Exemplo de Resposta:**
# ```json
# {
#   "partidos": [
#     {"partido": "Partido A", "percentual": 15.0},
#     {"partido": "Partido B", "percentual": 10.0}
#   ],
#   "concentração": "A Câmara apresenta uma distribuição com muitos partidos de baixa representatividade.",
#   "implicacoes": "A fragmentação pode dificultar a formação de coalizões majoritárias, aumentando a necessidade de negociações políticas."
# }

#     '''
# )



def coletar_serie_despesas_diárias_deputados(deputados_df, url_base="https://dadosabertos.camara.leg.br/api/v2"):
    """
    Coleta as informações das despesas dos deputados e salva em um arquivo Parquet.
    
    Args:
        deputados_df (pd.DataFrame): DataFrame contendo os IDs dos deputados.
        url_base (str): URL base da API da Câmara dos Deputados.

    Returns:
        None
    """
    despesas_list = []

    for deputado_id in deputados_df['id']:
        url = f"{url_base}/deputados/{deputado_id}/despesas"
        params = {'ano': 2023, 'ordem': 'ASC'}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            despesas = response.json().get('dados', [])
            for item in despesas:
                despesas_list.append({
                    'data': datetime.fromisoformat(item['dataDocumento']).date(),  
                    'deputado_id': deputado_id,
                    'tipo_despesa': item['tipoDespesa'],
                    'valor_liquido': item['valorLiquido']
                })
                

    # Convertendo a lista de despesas em DataFrame
    despesas_df = pd.DataFrame(despesas_list)
    if not despesas_df.empty:
        # Agrupando os dados
        despesas_agrupadas = despesas_df.groupby(['data', 'deputado_id', 'tipo_despesa']) \
            .sum(numeric_only=True).reset_index()

        # Salvando como arquivo Parquet
        despesas_agrupadas.to_parquet('data/serie_despesas_diárias_deputados.parquet', index=False)
        print("Arquivo salvo: data/serie_despesas_diárias_deputados.parquet")
    else:
        print("Nenhuma despesa coletada.")


deputados_df = pd.read_parquet('data/deputados.parquet')
coletar_serie_despesas_diárias_deputados(deputados_df)

# prompt = """
# Você é um analista de dados experiente. Abaixo está uma descrição dos dados disponíveis em um arquivo Parquet.

# **Dados Disponíveis:**
# Arquivo: `data/serie_despesas_diárias_deputados.parquet`
# - `data`: Data do registro da despesa.
# - `deputado_id`: ID do deputado.
# - `tipo_despesa`: Tipo da despesa registrada.
# - `valor_liquido`: Valor líquido da despesa.

# **Tarefa:**
# Escreva um código Python para realizar as seguintes análises:
# 1. Identificar os 5 deputados com maiores despesas totais no ano.
# 2. Analisar a média mensal das despesas por tipo de despesa.
# 3. Gerar uma série temporal mostrando a variação diária do total de despesas da câmara.

# Estruture o código de forma clara e utilizável.

# """



# # Carregar o arquivo Parquet
# try:
#     despesas = pd.read_parquet('data/serie_despesas_diárias_deputados.parquet')
# except FileNotFoundError:
#     print("Erro: Arquivo 'data/serie_despesas_diárias_deputados.parquet' não encontrado.")
#     exit()
# except Exception as e:
#     print(f"Erro ao carregar o arquivo: {e}")
#     exit()


# # 1. Top 5 deputados com maiores despesas totais no ano
# despesas['ano'] = pd.to_datetime(despesas['data']).dt.year
# despesas_por_deputado = despesas.groupby(['ano', 'deputado_id'])['valor_liquido'].sum().reset_index()

# # Considerando que o ano em questão está presente nos dados:
# ano_atual = despesas['ano'].max()  # Ou especifique o ano desejado, ex: ano_atual = 2023

# top_5_deputados = despesas_por_deputado[despesas_por_deputado['ano'] == ano_atual].nlargest(5, 'valor_liquido')       
# print("\nTop 5 deputados com maiores despesas no ano", ano_atual)
# print(top_5_deputados)



# # 2. Média mensal das despesas por tipo de despesa
# despesas['mes'] = pd.to_datetime(despesas['data']).dt.to_period('M')
# media_mensal_por_tipo = despesas.groupby(['mes', 'tipo_despesa'])['valor_liquido'].mean().reset_index()
# print("\nMédia mensal das despesas por tipo de despesa:")
# print(media_mensal_por_tipo)



# # 3. Série temporal da variação diária do total de despesas da câmara
# despesas_diarias = despesas.groupby('data')['valor_liquido'].sum()
# print("\nSérie temporal das despesas diárias:")
# print(despesas_diarias.head())

# # Plotar o gráfico da série temporal
# plt.figure(figsize=(12, 6))
# plt.plot(despesas_diarias.index, despesas_diarias.values)
# plt.xlabel('Data')
# plt.ylabel('Total de Despesas')
# plt.title('Variação Diária do Total de Despesas da Câmara')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# depesas_df = pd.read_parquet('data/serie_despesas_diárias_deputados.parquet')
# # Definir o conteúdo do prompt
# prompt = """
# Você é um modelo de linguagem treinado para gerar insights a partir de dados estruturados. Com base nas seguintes análises realizadas:

# 1. Top 5 deputados com maiores despesas no ano 2024
#       ano  deputado_id  valor_liquido
# 512  2024        74696      100000.00
# 535  2024       204489       25000.00
# 529  2024       178937       20800.00
# 538  2024       204553       18014.67
# 560  2024       220687        8400.00

# 2. Média mensal das despesas por tipo de despesa:
#         mes                                       tipo_despesa  valor_liquido
# 0   2023-02                      COMBUSTÍVEIS E LUBRIFICANTES.     585.223861
# 1   2023-02      CONSULTORIAS, PESQUISAS E TRABALHOS TÉCNICOS.   10400.000000
# 2   2023-02               DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR.    7215.628898
# 3   2023-02         FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR     113.690000
# 4   2023-02  HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO ...    1469.870000
# ..      ...                                                ...            ...
# 71  2024-01      LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES    3233.330000
# 72  2024-01  MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE ...    2364.424468
# 73  2024-02               DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR.    5000.000000
# 74  2024-02  MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE ...    1313.722500
# 75  2024-03  MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE ...     514.070000

# [76 rows x 3 columns]

# 3. Série temporal das despesas diárias:
# data
# 2023-02-01    454652.46
# 2023-02-02    197433.75
# 2023-02-03    148287.89
# 2023-02-04      2886.05
# 2023-02-05     74115.81

# Com base nos resultados dessas análises, por favor, gere insights relevantes que podem ajudar na tomada de decisão. Exemplos de insights que você pode produzir incluem:
# - Identificação de padrões de gastos.
# - Comportamentos estranhos ou anomalias nos gastos de deputados ou tipos de despesa.
# - Possíveis ações para controle ou otimização de despesas.

# As informações para análise estão armazenadas nos seguintes dados:
# - Arquivo: {despesas_df}
# - Variáveis: `data`, `deputado_id`, `tipo_despesa`, `valor_liquido`.

# """
# # Chave de API do OpenAI
# api_key = os.getenv('OPENAI_API_KEY')
# genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# model = genai.GenerativeModel('gemini-1.5-pro')
# response = model.generate_content(prompt)
# print(response.text)


# import requests
# import pandas as pd
# import os

def coletar_proposicoes_por_numero(numeros: list, max_proposicoes: int = 10):
    """
    Coleta informações sobre proposições com os números fornecidos da API da Câmara dos Deputados.
    Salva os dados em um arquivo Parquet.
    
    Parâmetros:
        numeros (list): Lista de números das proposições a serem coletadas (ex.: [40, 46, 62]).
        max_proposicoes (int): Número máximo de proposições por número.
    """
    base_url = "https://dadosabertos.camara.leg.br/api/v2/proposicoes"
    headers = {"accept": "application/json"}
    all_propositions = []

    for numero in numeros:
        page = 1
        collected = 0
        while collected < max_proposicoes:
            params = {
                "numero": numero,
                "itens": 50,  # Máximo por página para otimizar requisições
                "pagina": page,
                "ordem": "asc",
                "ordenarPor": "id"
            }

            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"Erro ao coletar dados para o número {numero}: {response.status_code}")
                break

            data = response.json().get("dados", [])
            if not data:  # Se não houver dados, sair do loop
                break

            for item in data:
                all_propositions.append({
                    "id": item.get("id"),
                    "uri": item.get("uri"),
                    "siglaTipo": item.get("siglaTipo"),
                    "codTipo": item.get("codTipo"),
                    "numero": item.get("numero"),
                    "ano": item.get("ano"),
                    "ementa": item.get("ementa")
                })
                collected += 1
                if collected >= max_proposicoes:
                    break

            page += 1

    # Converter os dados em DataFrame
    df = pd.DataFrame(all_propositions)


    output_dir = "data"
    

    # Salvar os dados no formato Parquet
    output_path = os.path.join(output_dir, "proposicoes_deputados.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Dados salvos em {output_path}")




import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class ProposicaoSummarizer():
    def __init__(self, model_name, apikey, parquet_path, window_size, overlap_size, system_prompt, generation_config=None):
        self.parquet_path = parquet_path
        self.data = pd.read_parquet(parquet_path)
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.chunks = self.__data_to_chunks()
        self.model = self.__create_model(apikey, model_name, system_prompt, generation_config)

    def __create_model(self, apikey, model_name, system_prompt, generation_config=None):
        genai.configure(api_key=os.getenv(apikey))
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        if generation_config is None:
            generation_config = {
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 20,
                'max_output_tokens': 1000
            }
        return genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    def __data_to_chunks(self):
        """
        Divide as proposições em chunks baseados em 'ementa'.
        """
        ementas = self.data['ementa'].tolist()
        n = self.window_size  # Tamanho de cada chunk
        m = self.overlap_size  # Sobreposição entre chunks
        return [ementas[i:i+n] for i in range(0, len(ementas), n-m)]

    def __create_chunk_prompt(self, chunk, prompt_user):
        """
        Cria um prompt para o modelo de linguagem com base no chunk.
        """
        episode_lines = '\n'.join(chunk)
        prompt = f"""
        Summarize the following set of propositions considering the user's request:
        # USER
        {prompt_user}
        # OUTPUT INSTRUCTION
        The summary output must be written as a plain JSON with a field 'summary'.
        ###### PROPOSICOES
        {episode_lines}
        ######
        Summarize them.
        """
        return prompt

    def __summarize_chunks(self, prompt_user):
        """
        Realiza a sumarização para cada chunk.
        """
        chunk_summaries = []
        for i, chunk in enumerate(self.chunks):
            print(f'Summarizing chunk {i+1} of {len(self.chunks)}')
            prompt = self.__create_chunk_prompt(chunk, prompt_user)
            response = self.model.generate_content(prompt)
            chunk_summaries.append(response.text)
        return chunk_summaries

    def summarize(self, prompt_user):
        """
        Realiza a sumarização geral das proposições.
        """
        print('Summarizing all propositions...')
        self.chunk_summaries = self.__summarize_chunks(prompt_user)
        summaries = [f"- {x}\n" for x in self.chunk_summaries]
        prompt_summary = f"""
        # User: {prompt_user} 
        ### chunk summaries
        {summaries}
        ###

        Based on the chunk summaries, generate a final summary in JSON format with a field 'assistant'.
        """
        print('Generating final summary...')
        response = self.model.generate_content(prompt_summary)
        return response.text


# Exemplo de uso
if __name__ == "__main__":
    summarizer = ProposicaoSummarizer(
        model_name="gemini-1.5-flash",
        apikey="GEMINI_API_KEY",
        parquet_path="data/proposicoes_deputados.parquet",
        window_size=5,
        overlap_size=2,
        system_prompt="You are an expert summarizer of legislative propositions.",
        generation_config={
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 1500
        }
    )
    
    prompt_user = "Provide a concise summary of the propositions, highlighting their key purposes."
    final_summary = summarizer.summarize(prompt_user)
    print(final_summary)
