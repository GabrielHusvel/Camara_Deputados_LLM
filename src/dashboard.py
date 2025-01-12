
import streamlit as st
import pandas as pd
import yaml
import json
from PIL import Image
import plotly.express as px
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import openai




def carregar_dados(caminho):
    try:
        if caminho.endswith('.json'):
            with open(caminho, encoding='utf-8') as f:  
                return json.load(f)
        elif caminho.endswith('.parquet'):
            return pd.read_parquet(caminho)
        else:
            raise ValueError("Formato de arquivo não suportado.")
    except Exception as e:
        st.error(f"Erro ao carregar dados do arquivo {caminho}: {e}")
        return None

# Ddos de configuração
try:
    with open("data/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    config = {}  
    st.error("Arquivo de configuração 'data/config.yaml' não encontrado.")

# Insights de distribuição de deputados
try:
    with open("data/insights_distribuicao_deputados.json", "r", encoding="utf-8") as file:
        insights_distribuicao = json.load(file)
except FileNotFoundError:
    insights_distribuicao = {}  
    st.error("Arquivo de insights 'data/insights_distribuicao_deputados.json' não encontrado.")


# Dados de deputados
try:
    deputados_df = pd.read_parquet("data/deputados.parquet")
except FileNotFoundError:
    deputados_df = pd.DataFrame()  
    st.error("Arquivo de dados 'data/deputados.parquet' não encontrado.")

# Carregar a imagem
try:
    distribuicao_deputados_image = Image.open("data/distribuicao_deputados.png")
except FileNotFoundError:
    distribuicao_deputados_image = None
    st.error("Imagem 'data/distribuicao_deputados.png' não encontrada.")


# Layout do Streamlit
st.set_page_config(page_title="Dashboard Deputados", layout="wide")

# Abas
tabs = st.tabs(["Overview", "Despesas", "Proposições"])



with tabs[0]:
    st.title("Visão Geral dos Deputados")
    st.write("""
    Esta seção fornece uma visão geral dos dados disponíveis na API dos Deputados. 
    A API oferece informações sobre deputados, incluindo seus perfis, despesas, proposições, votações, e muito mais.  
    Explore as outras abas para análises mais detalhadas sobre despesas e proposições.

    **Exemplos de informações disponíveis na API:**

    * **Dados do Deputado:** Nome, partido, estado, foto, etc.
    * **Despesas Parlamentares:** Detalhes sobre gastos com verba indenizatória.
    * **Proposições:** Leis propostas, emendas, relatórios, etc.
    * **Votações:** Registro de votos em diferentes proposições.

    Recursos adicionais e documentação da API podem ser encontrados no site da Câmara dos Deputados.
     """)
    st.write("Dados obtidos da API dos Deputados.")

    if distribuicao_deputados_image:
        st.image(distribuicao_deputados_image, caption="Distribuição dos Deputados")

    if config:
        for key, value in config.items():
             st.write(f" {value}")
  
    if insights_distribuicao:
        st.write("Insights da Distribuição dos Deputados:")
        st.json(insights_distribuicao)
        
 

with tabs[1]:
    st.title("Despesas Parlamentares")

    # Carregar dados
    insights_despesas = carregar_dados("data/insights_despesas_deputados.json")
    serie_despesas_diarias = carregar_dados("data/serie_despesas_diárias_deputados.parquet")

    # Exibir insights
    if insights_despesas:
        for insight in insights_despesas.values():
            st.write(f"- {insight}")
    else:
        st.warning("Não foi possível carregar os insights das despesas.")

    # Selecionar deputado
    if serie_despesas_diarias is not None :
        deputados = serie_despesas_diarias["deputado_id"].unique()

    deputado_selecionado = st.selectbox("Selecione um deputado:", deputados)

    # Filtrar e exibir dados
    if serie_despesas_diarias is not None and "deputado_id" in serie_despesas_diarias:
        despesas_selecionadas = serie_despesas_diarias[
            serie_despesas_diarias["deputado_id"] == deputado_selecionado
        ]
    else:
        despesas_selecionadas = pd.DataFrame()

    if not despesas_selecionadas.empty:
        fig = px.bar(despesas_selecionadas, x="data", y="valor_liquido", title=f"Despesas diárias de {deputado_selecionado}")
        st.plotly_chart(fig)
    else:
        st.warning(f"Não há dados de despesas para {deputado_selecionado}.")


with tabs[2]:
    st.title("Proposições Legislativas")

    proposicoes = carregar_dados("data/proposicoes_deputados.parquet")
    sumarizacao_proposicoes = carregar_dados("data/sumarizacao_proposicoes.json") 

    st.write(proposicoes)
    st.write(sumarizacao_proposicoes)

    import pandas as pd
    import json
    import faiss
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import os
    import yaml
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import AIMessage, HumanMessage
    msgs = StreamlitChatMessageHistory()


    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(messages=msgs, memory_key="chat_history", return_messages=True)

    memory = st.session_state.memory

    # 1. Carregar os Dados
    def load_data():
        deputados = carregar_dados("data/deputados.parquet")
        count_by_party = deputados.groupby("siglaPartido").size()
        despesas = carregar_dados("data/serie_despesas_diárias_deputados.parquet")
        insights_dist = carregar_dados("data/insights_distribuicao_deputados.json")
        insights_desp = carregar_dados("data/insights_despesas_deputados.json")

        

        
        with open("data/config.yaml") as f:
            config = yaml.safe_load(f)
        
        return deputados, despesas, proposicoes, insights_dist, insights_desp, sumarizacao_proposicoes, count_by_party, config

    # 2. Extrair Textos
    def extract_texts(deputados, despesas, proposicoes, insights_dist, insights_desp, sumarizacao_proposicoes):
        texts = []

        # Deputados
        for index, dep in deputados.iterrows():
            texts.append(f"Deputado: {dep['nome']} ID: {dep['id']} Partido: {dep['siglaPartido']}  Uf: {dep['siglaUf']}")

        # Despesas
        for index, desp in despesas.iterrows():
            texts.append(f"Despesa: {desp['tipo_despesa']} ID: {desp['deputado_id']} Valor: {desp['valor_liquido']}  Data: {desp['data']}")

        # Proposições
        for index, prop in proposicoes.iterrows():
            if prop['numero'] == 40:
                texts.append(f"Proposição: {prop['ementa']}  Partido: {prop['siglaTipo']}  Ano: {prop['ano']} Id: {prop['id']} Tema: Economia")
                    
            if prop['numero'] == 46:
                texts.append(f"Proposição: {prop['ementa']}  Partido: {prop['siglaTipo']}  Ano: {prop['ano']} Id: {prop['id']} Tema: Educacao")
             
            if prop['numero'] == 62:
                texts.append(f"Proposição: {prop['ementa']}  Partido: {prop['siglaTipo']}  Ano: {prop['ano']} Id: {prop['id']} Tema: Ciencia, Tecnologia e Inovacao")
        texts.extend(insights_dist)
        texts.extend(insights_desp)

        # Sumarização Proposições
        for key, value in sumarizacao_proposicoes.items():
            texts.append(f"Sumarização ({key}): {value}")

        return texts

    # 3. Vetorização
    def vectorize_texts(texts):
        model_name = 'all-MiniLM-L6-v2'
        embedding_model = SentenceTransformer(model_name, device='cuda')
        embeddings = embedding_model.encode(texts).astype("float32")
        
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        
        return index, embedding_model

    # 4. Responder Perguntas
    def answer_query(query, index, embedding_model, texts, model, count_by_party):
        query_embedding = embedding_model.encode([query]).astype("float32")
        distances, indices = index.search(query_embedding, 500)

        db_text = '\n'.join([f"- {texts[indices[0][i]]}" for i in range(500)])
        
        prompt = f"""
        Respond to the <user question> considering the information retrieved from the <database>.
        Read several lines to try to understand how to respond it. 
        Ensure the response is a valid JSON object with a 'response' field. Avoid any extra characters.
        ##
        <user question>
        {query}
        
        ##)
        <database>
        {db_text}
        {count_by_party}
        ##
        The response must be formatted as a JSON with the field 'response'.
        """
        
        responses = [model.generate_content(prompt) for _ in range(1)]
        responses = [r.text.replace("```json\n",'').replace("\n```",'') for r in responses]
        try:
            responses = [json.loads(r)['response'] for r in responses]
            return responses
        except:
            return responses


    # Configurar o modelo generativo
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        "gemini-pro",
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        generation_config={
            'temperature': 0.1,
        }
    )

    # Ddos e configurar modelo
    deputados, despesas, proposicoes, insights_dist, insights_desp, sumarizacao, count_by_party, config = load_data()
    texts = extract_texts(deputados, despesas, proposicoes, insights_dist, insights_desp, sumarizacao)
    index, embedding_model = vectorize_texts(texts)

    # Memória
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    # Pergunta do Usuário
    query = st.chat_input("Faça sua pergunta sobre os deputados:")

    if query:
        chat_history = st.session_state["memory"].chat_memory.messages
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(f"{msg.content}")
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(f"{msg.content}")
                    
        with st.spinner("Agent is responding..."):
            response = answer_query(query, index, embedding_model, texts, model, count_by_party)
            st.session_state['conversation'].append({'user': query, 'response': response[0]})

    # Histórico da Conversa
    for message in st.session_state['conversation']:
        st.write(f"**Usuário:** {message['user']}")
        st.write(f"**Resposta:** {message['response']}")
        



