# Análise de Dados da Câmara dos Deputados com Streamlit e LLM

Este aplicativo Streamlit oferece uma interface interativa para explorar dados da Câmara dos Deputados, incluindo informações sobre deputados, suas despesas e proposições.  Ele utiliza um modelo de linguagem grande (LLM) para responder a perguntas com base nesses dados, fornecendo insights rápidos e acessíveis.

O aplicativo é dividido em três abas principais:

## Como usar

1. **Clone o repositório:**

Instale as dependências:

pip install -r requiements.txt

Rode o app

streamlit run app.py
Navegue pelas abas: Use a barra lateral para alternar entre as abas "Overview", "Despesas" e "Proposições".
Explore os dados: Utilize os gráficos e dataframes interativos para analisar as informações.
Faça perguntas ao LLM: Na aba "Proposições", insira suas perguntas no chat e receba respostas informativas.

Limitações
O LLM é treinado com os dados disponíveis e pode não ter conhecimento de informações externas.
A precisão das respostas do LLM depende da qualidade e abrangência dos dados.
O aplicativo pode ter limitações de desempenho dependendo do tamanho dos dados e da complexidade das consultas.
