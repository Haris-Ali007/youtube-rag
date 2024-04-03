import os
from dotenv import load_dotenv
from pytube import YouTube
import tempfile
import whisper
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

if __name__=="__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    transcription_file = 'transcription.txt'

    ## loading model ##
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    whisper_model = whisper.load_model("base")
    embeddings_model = OpenAIEmbeddings()


    ## loading text from the youtube video ##
    if not os.path.isfile(transcription_file): # set this on an if that will tell if it is for new video
        video_url = "https://www.youtube.com/watch?v=-BjWCx-50Lc"
        yt_obj = YouTube(video_url)
        audio = yt_obj.streams.filter(only_audio=True).first()
        with tempfile.TemporaryDirectory() as fileObj:
            file = audio.download(fileObj)
            transcription = whisper_model.transcribe(file, fp16=False)['text'].strip()

        with open(transcription_file, 'w') as fp:
            fp.write(transcription)

    ## defining prompt template ##
    template = """
    Answer questions using the context below. If no answer
    found reply 'Sorry no relevant information found'
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    ## storing data into vector store ##
    document_loader = TextLoader('transcription.txt')
    documents = document_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    split_documents = splitter.transform_documents(documents)

    # setting up vector store
    vectore_store = PineconeVectorStore.from_documents(
                                    documents,
                                    embeddings_model,
                                    index_name="demo-app"
    )
    parser = StrOutputParser()
    retriever = vectore_store.as_retriever()
    chain = (
        {'context': retriever, 'question':RunnablePassthrough()}
        | prompt | model | parser
    )

    answer = chain.invoke("How to build a parser")
    print(answer)