import os
import base64
from io import BytesIO

import fitz
import uuid
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_chroma import Chroma


load_dotenv()

# Configuration
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
LLM_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

st.set_page_config(
    page_title="Multimodal RAG",
)

st.title("NVIDIA Multimodal RAG")
st.caption("Ask questions about PDF documents and images using multimodal AI.")

# NVIDIA API Key
try:
    api_key = os.getenv("NVIDIA_API_KEY") or st.secrets["NVIDIA_API_KEY"]
except Exception:
    api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("Please configure your NVIDIA API key.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


# Multimodal Embeddings
class NemotronEmbeddings(Embeddings):

    def embed_documents(self, texts):
        response = client.embeddings.create(
            input=texts,
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={
                "modality": "text",
                "input_type": "passage"
            }
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = client.embeddings.create(
            input=[text],
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={
                "modality": "text",
                "input_type": "query"
            }
        )
        return response.data[0].embedding


# Vision LLM
llm = ChatNVIDIA(
    model=LLM_MODEL,
    api_key=api_key,
    temperature=0.2,
    max_tokens=4096
)


# Convert uploaded document into page images
def process_document(file):

    file_bytes = file.getvalue()

    if file.type == "application/pdf":

        pdf = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        images = []

        for page in pdf:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(1.3, 1.3),
                alpha=False
            )

            images.append(
                pix.tobytes("jpeg")
            )

        pdf.close()

    else:

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)

        images = [buffer.getvalue()]

    return images


# Generate image embeddings and create ChromaDB
def create_vector_database(images):

    embeddings = NemotronEmbeddings()

    image_embeddings = []
    metadatas = []
    page_images = []

    progress = st.progress(0)

    # Step 1: Generate image embeddings
    for index, image_bytes in enumerate(images):

        image_b64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        page_images.append(image_b64)

        response = client.embeddings.create(
            input=[
                f"data:image/jpeg;base64,{image_b64}"
            ],
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={
                "modality": "image",
                "input_type": "passage"
            }
        )

        vector = response.data[0].embedding

        image_embeddings.append(vector)

        metadatas.append({
            "page": index + 1
        })

        progress.progress(
            (index + 1) / len(images)
        )

    # Step 2: Initialize ChromaDB
    vectorstore = Chroma(
        collection_name=f"multimodal_rag_{uuid.uuid4().hex}",
        embedding_function=embeddings
    )

    # Step 3: Add precomputed image embeddings
    vectorstore._collection.add(
        ids=[
            f"page-{i+1}" for i in range(len(images))
        ],
        embeddings=image_embeddings,
        documents=[
            f"Page {i+1}" for i in range(len(images))
        ],
        metadatas=metadatas
    )

    # Step 4: Store in Streamlit session
    st.session_state.vectorstore = vectorstore
    st.session_state.page_images = page_images

    st.success(
        f"Successfully processed {len(images)} page(s)!"
    )


# Document Upload
uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "webp"]
)

if st.button("Process Document", type="primary"):

    if uploaded_file is None:
        st.warning("Please upload a document first.")

    else:
        try:
            with st.spinner("Processing document..."):

                images = process_document(uploaded_file)

                create_vector_database(images)

        except Exception as e:
            st.error(f"Processing failed: {e}")


# Question Answering
if "vectorstore" in st.session_state:

    st.divider()

    question = st.text_input(
        "Ask a question about your document"
    )

    if question:

        try:
            with st.spinner("Finding relevant pages and generating answer..."):

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

                retrieved_docs = retriever.invoke(question)

                content = [
                    {
                        "type": "text",
                        "text": (
                            "Answer the user's question using the "
                            "provided document page images. "
                            "Explain charts, tables, diagrams, "
                            "and visual information when relevant. "
                            "Do not invent information. "
                            "If the answer is not available in the "
                            "provided pages, clearly say so.\n\n"
                            f"Question: {question}"
                        )
                    }
                ]

                for doc in retrieved_docs:

                    page_number = doc.metadata["page"]

                    image_b64 = st.session_state.page_images[
                        page_number - 1
                    ]

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,{image_b64}"
                            )
                        }
                    })

                message = HumanMessage(content=content)

                response = llm.invoke([message])

            st.subheader("Answer")
            st.write(response.content)

            with st.expander("View Retrieved Pages"):

                for doc in retrieved_docs:

                    page_number = doc.metadata["page"]

                    st.write(f"Page {page_number}")

                    st.image(
                        base64.b64decode(
                            st.session_state.page_images[
                                page_number - 1
                            ]
                        ),
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Unable to generate answer: {e}")
