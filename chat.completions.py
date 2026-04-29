from openai import OpenAI
import re
import pandas as pd
from tqdm import tqdm

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="fake-key"
)

def classify_message(message):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[
            {
            "role": "system",
            "content": """
            Você é um classificador de desinformação em saúde, classificando uma base de mensagens do whatsapp enviadas no brasil coletadas no periodo de 2020 a 2023. Você pode escolher mais do que uma classificação caso seja necessário e pertinente, sendo assim, classifique mensagens como:
            0 Não é desinformação: Mensagens informativas ou que não possuam desinformação
            1.1 Evidência anedótica: Uso de relatos pessoais no lugar de dados científicos
            1.2 Pesquisa Ultrapassada: Uso de pesquisas antigas ou já comprovadas como falsas
            1.3 Má Qualidade de Pesquisa: Reclamações sobre insuficiência ou a qualidade de estudos
            1.4 Conhecimento Falível: Crença de que nunca se pode ter certeza sobre ciência
            2.1 Estudos Fabricados: Criação de estudos científicos falsos
            2.2 Reportamento Seletivo: Seleção de dados para apoiar uma narrativa e aplicar resultados fora do contextocorreto
            2.3 Conspiração: Alegação de intenções ocultas e maléficas em ações de saúde
            2.4 Censura: Alegações de que informações contrárias são suprimidas
            3.1 Produtos/Terapias Alternativas: Promoção de tratamentos sem validação científica e venda de produtos de saúdecom alegações falsas
            3.2 Alternativas à Vacinação: Alegações de que medidas ou tratamentos substituem vacinas
            4.1 Alarmismo: Disseminação de medos exagerados sobre tratamentos ou doenças. (Alegaçõessobre toxicidade de componentes de vacinas/tratamentos)
            4.2 Afirmações Pseudocientíficas: Afirmações que soam científicas, mas não são, utilizando de termos de poucoconhecimento público para fingir conhecimento
            4.3 Confundindo correlação com causalidade: Confundir correlação com causalidade
            4.4 Má Conduta Financeira: Ganância financeira como motor de decisões em saúde
            4.5 Direito de Autonomia: Defesa do direito individual de recusar tratamentos
            5.1 Proteção Imperfeita: Argumentos de que vacinas/tratamentos não são 100% eficazes
            5.2 Imunidade de Rebanho: Argumentos de que a proteção dos outros já basta
            5.3 Imunidade Natural: Defesa da imunidade adquirida naturalmente em vez de vacinação
            5.4 Riscos Insuficientes: Dizer que doenças são pouco graves ou não causam qualquer problema,minimizando a necessidade de vacinação
            6.1 Transmissão Direta: Medo de contrair doença através da vacinação/tratamento
            6.2 Efeitos Colaterais Específicos: Relatos de efeitos adversos específicos
            6.3 Aplicação Perigosa/Incompetência: Problemas relacionados à forma de administração ou negligência na aplicação porparte de autoridades de saúde
            6.4 Indivíduos de Alto Risco: Elegem grupos supostamente mais vulneráveis a riscos relacionados a tratamentosmédicos e exageram nas exclamações (crianças, idosos, etc)
            7.1 Mitos Virais: Mitos de saúde amplamente disseminados online
            7.2 Crenças Religiosas e Éticas: Conflito entre vacinação/tratamento e crenças pessoais
            A resposta devem ser as categorias (na formatação: número nome) divididas por virgula
             """
            },
            {
            "role": "user",
            "content": message,
            }
        ]
    )

    text = response.choices[0].message.content
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    print(result)
    return result.strip()

tqdm.pandas()

df = pd.read_csv("dataset_950.csv")
df["classificacao"] = df["Mensagem"].progress_apply(classify_message)
df.to_csv("resultado.csv", index=False)
