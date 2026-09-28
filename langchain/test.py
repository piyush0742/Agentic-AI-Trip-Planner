from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    project="agentic-ai-510016",
    vertexai=True,
)

result = embeddings.embed_query("Golden Gate Bridge")
print(len(result))