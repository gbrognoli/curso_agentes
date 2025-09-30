import streamlit as st
import pandas as pd
# Importação do LangChain atualizada para o Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import matplotlib
matplotlib.use('agg') # Backend para renderizar gráficos sem GUI
import matplotlib.pyplot as plt
import io

# --- FUNÇÃO DO AGENTE (Atualizada para Google Gemini) ---
def setup_agent(df, api_key):
    """Cria e configura o agente LangChain para interagir com o DataFrame usando Google Gemini."""
    # Inicializa o LLM do Google Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0)
    
    # Retorna o agente, agora com o LLM do Gemini
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        allow_dangerous_code=True # Adicionado para compatibilidade com versões recentes
    )

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (com Gemini)")
st.write("Faça o upload de um arquivo CSV e comece a fazer perguntas!")

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
# Usado para guardar o histórico da conversa
if "messages" not in st.session_state:
    st.session_state.messages = []
# Usado para guardar o agente e evitar recriá-lo a cada interação
if "agent" not in st.session_state:
    st.session_state.agent = None

# --- SIDEBAR PARA CONFIGURAÇÃO ---
with st.sidebar:
    st.header("Configuração")
    # Campo atualizado para a chave da API do Google
    google_api_key = st.text_input("Insira sua Chave da API do Google", type="password")
    
    # Campo para upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if st.button("Iniciar Agente"):
        if not google_api_key:
            st.warning("Por favor, insira sua chave da API do Google.")
        elif uploaded_file is None:
            st.warning("Por favor, faça o upload de um arquivo CSV.")
        else:
            with st.spinner("Carregando dados e configurando o agente..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.agent = setup_agent(df, google_api_key)
                    st.success("Agente pronto! Você já pode fazer perguntas.")
                    # Limpa o histórico de chat ao iniciar um novo agente
                    st.session_state.messages = [] 
                except Exception as e:
                    st.error(f"Ocorreu um erro ao carregar o arquivo: {e}")

# --- INTERFACE DE CHAT ---
# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output_type" in message and message["output_type"] == "plot":
            st.image(message["content"])
        else:
            st.markdown(message["content"])

# Captura a pergunta do usuário
if prompt := st.chat_input("Faça uma pergunta sobre seus dados..."):
    # Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera e exibe a resposta do agente
    with st.chat_message("assistant"):
        if st.session_state.agent is None:
            st.warning("Por favor, configure o agente na barra lateral primeiro.")
        else:
            with st.spinner("O agente está pensando..."):
                try:
                    # Captura a saída padrão para encontrar gráficos
                    plt.figure() # Inicia uma nova figura para o gráfico
                    response = st.session_state.agent.run(prompt)
                    
                    # Salva a figura atual (se houver) em um buffer de memória
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    
                    # Verifica se o buffer contém uma imagem (se algo foi plotado)
                    if buf.getbuffer().nbytes > 1000: # Heurística simples
                        st.image(buf, caption="Gráfico gerado pelo agente")
                        st.session_state.messages.append({"role": "assistant", "content": buf, "output_type": "plot"})
                    else:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")
