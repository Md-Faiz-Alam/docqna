# from docqna.generation.providers import OllamaProvider
# from docqna.settings import get_settings


# settings = get_settings()

# provider = OllamaProvider(
#     base_url=settings.ollama_base_url,
#     model=settings.ollama_model,
# )

# print("Provider:", settings.llm_provider)
# print("Model:", settings.ollama_model)
# print("Ollama available:", provider.health_check())

#####

# from docqna.generation.providers import OllamaProvider
# from docqna.settings import get_settings


# settings = get_settings()

# provider = OllamaProvider(
#     base_url=settings.ollama_base_url,
#     model=settings.ollama_model,
# )

# response = provider.generate(
#     prompt="Reply with exactly: DocQ&A Ollama provider working."
# )

# print(response)

###

# from docqna.generation.providers import OllamaProvider
# from docqna.settings import get_settings


# settings = get_settings()

# provider = OllamaProvider(
#     base_url=settings.ollama_base_url,
#     model=settings.ollama_model,
# )

# response = provider.generate(
#     system_prompt=(
#         "You are a document question-answering assistant. "
#         "Answer concisely."
#     ),
#     prompt="What does RAG stand for?",
# )

# print(response)

###

# from docqna.generation.providers import OllamaProvider


# provider = OllamaProvider(
#     base_url="http://localhost:9999",
#     model="phi3:mini",
# )

# print(provider.health_check())

###

