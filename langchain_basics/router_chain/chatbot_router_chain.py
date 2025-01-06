from langchain_openai import OpenAI

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

from dotenv import dotenv_values

config = dotenv_values("/Users/manilsonlussati/Documents/workspace/langchain-exploring/.env")

llm = OpenAI(temperature=0.0, openai_api_key=config["OPENAI_KEY"])

# Definindo os templates de prompt para os destinos
conta_corrente_template = """Você é um especialista em contas correntes de um banco. \
Ajude um cliente com dúvidas relacionadas ao saldo, extrato e movimentações da conta. \
Forneça informações precisas e orientações sobre como resolver problemas, \
como transações não reconhecidas ou ajustes no saldo.

Aqui está a pergunta:
{input}"""

emprestimos_template = """Você é um especialista em empréstimos bancários. \
Ajude um cliente com dúvidas sobre taxas de juros, condições de pagamento, \
ou o processo de solicitação de empréstimos. Ofereça orientações claras e suporte \
para ajudá-lo a tomar decisões financeiras informadas.

Aqui está a pergunta:
{input}"""

suporte_tecnico_template = """Você é um especialista em suporte técnico para serviços bancários digitais. \
Ajude um cliente que está enfrentando problemas ao acessar ou utilizar serviços digitais, \
como aplicativos móveis ou internet banking. Forneça instruções detalhadas para resolver \
problemas técnicos e garanta que o cliente se sinta apoiado durante todo o processo.

Aqui está a pergunta:
{input}"""

prompt_infos = [
    {
        "name": "Conta Corrente",
        "description": "Bom para responder perguntas relacionadas à conta corrente do cliente.",
        "prompt_template": conta_corrente_template
    },
    {
        "name": "Empréstimos",
        "description": "Bom para responder perguntas relacionadas a empréstimos bancários.",
        "prompt_template": emprestimos_template
    },
    {
        "name": "Suporte Técnico",
        "description": "Bom para responder perguntas relacionadas a problemas técnicos com serviços bancários.",
        "prompt_template": suporte_tecnico_template
    }
]

# Criando os destinos
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Criando o chain padrão
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Definindo o template do roteador
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

# Criando o roteador
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Combinando os chains
final_chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain, verbose=True
                        )

# Executando o chain combinado
print(final_chain.run("Eu preciso saber meu saldo atual e entender por que uma transação foi cobrada duas vezes."))
