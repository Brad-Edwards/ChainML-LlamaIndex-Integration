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
