# Youtube-rag
This repository is a simple RAG system built to answer questions from a youtube video.

# How to setup:
* Simply clone the repo
* Install the required libraries
* Create a **.env** file and add these two environment variables:
  ```
  OPENAI_API_KEY=""
  PINECONE_API_KEY=""
  ```
  Copy your keys in it.
* Setup free pinecone account. Create a simple project and in that project create an index. This name will be provided at the input.

# Running app:
`python rag.py --url {video_url} --index {index-name} --model {model name default: gpt-3.5-turbo} --transcript {path to create transcript file}`

Below is a demo of output:
![image](https://github.com/Haris-Ali007/youtube-rag/assets/54216004/05686ca3-c78b-4b02-a81d-46053cbf5c78)
