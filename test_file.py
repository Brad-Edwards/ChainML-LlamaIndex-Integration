from council.filters import BasicFilter
from council.utils import Option
from council.evaluators import BasicEvaluator
from council.controllers import BasicController
from council.prompt import PromptBuilder
from council.skills import PromptToMessages, LLMSkill
from council.llm import OpenAILLMConfiguration, OpenAILLM
from council.agents import Agent
from council.chains import Chain
from council.contexts import SkillContext, ChatMessage
from council.skills import SkillBase
from llama_index.indices.query.base import BaseQueryEngine


class LlamaIndexSkill(SkillBase):
    queryEngine: BaseQueryEngine

    def __init__(self, query_engine: BaseQueryEngine):
        SkillBase.__init__(self, "llama index skill")
        self.queryEngine = query_engine

    def execute(self, context: SkillContext) -> ChatMessage:
        prompt = context.chat_history.try_last_user_message.unwrap(
            "no user message"
        ).message
        print(prompt)
        response = self.queryEngine.query(prompt)
        return self.build_success_message(response)


# agent to use index to provide context for more complex answers
import dotenv
import os

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY", None) is not None)

dotenv.load_dotenv()
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os
import requests

url = (
    "https://github.com/jerryjliu/llama_index/blob/main/examples/gatsby/gatsby_full.txt"
)
filename = url.split("/")[-1]

os.makedirs("gatsby_download", exist_ok=True)

response = requests.get(url)
with open(os.path.join("gatsby_download", filename), "wb") as f:
    f.write(response.content)

# build index of book / This step could be slow
documents = SimpleDirectoryReader("gatsby_download").load_data()
index = VectorStoreIndex.from_documents(documents)


query_engine = index.as_query_engine()

config = OpenAILLMConfiguration.from_env()
config.model = Option("gpt-3.5-turbo")
llm = OpenAILLM(config)

context_prompt = PromptToMessages(
    PromptBuilder(
        "Please identify query terms to respond to the following user request {{chat_history.last_message}}"
    )
)
context_query_skill = LLMSkill(
    llm,
    "You are an expert in the Great Gatbsy. Identify relevant query terms to search for context in the book.",
    context_messages=context_prompt.to_user_message,
)

index_skill = LlamaIndexSkill(query_engine)
index_prompt = PromptToMessages(
    PromptBuilder(
        "Here are relevant quotes from the book: {{chain_history.last_message}} \nUse this to respond to the following user request {{chat_history.last_message}}"
    )
)
response_skill = LLMSkill(
    llm,
    "You are an expert in the Great Gatbsy. Provide a helpful response to the user's question",
    context_messages=index_prompt.to_user_message,
)

chain = Chain(
    "docindex", "document index", [context_query_skill, index_skill, response_skill]
)
agent = Agent(BasicController([chain]), BasicEvaluator(), BasicFilter())
