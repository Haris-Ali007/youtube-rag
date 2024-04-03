import os
from dotenv import load_dotenv
from pytube import YouTube
import tempfile
import whisper
import argparse
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="Url for the youtube video")
    parser.add_argument("--index", type=str, help="Pinecone index for storing vector data")
    parser.add_argument("--model", type=str, nargs='?', default="gpt-3.5-turbo", help="LLM model name")
    parser.add_argument("--transcript", type=str, help="Path to transcript file")

    args = parser.parse_args()

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    transcription_file = args.transcript
    video_url = args.url

    ## loading model ##
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model=args.model)
    whisper_model = whisper.load_model("base")
    embeddings_model = OpenAIEmbeddings()
    ## loading text from the youtube video ##
    if not os.path.isfile(transcription_file): # set this on an if that will tell if it is for new video
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
                                    index_name=args.index
    )
    parser = StrOutputParser()
    retriever = vectore_store.as_retriever()
    chain = (
        {'context': retriever, 'question':RunnablePassthrough()}
        | prompt | model | parser
    )

    try:
        while True:
            question = input(">>> Enter your question \n")
            answer = chain.invoke(question)
            print(answer)
            print('********************************')
    except KeyboardInterrupt:
        print("Program exited")