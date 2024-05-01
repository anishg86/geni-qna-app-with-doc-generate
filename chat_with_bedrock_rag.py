import os
import yaml
import json
import streamlit as st
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS


def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
        data = loader.load()
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
        data = loader.load()
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file,encoding="utf-8")
        data = loader.load()
    elif extension in ['.csv','.xlsx'] :
        from langchain.document_loaders import CSVLoader
        print(f'Loading {file}')
        loader = CSVLoader(file)
        data = loader.load()
    elif extension == '.yaml' :
        from langchain.schema.document import Document
        print(f'Loading {file}')
        with open(file, 'r') as f:
            json_string = json.dumps(yaml.safe_load(f))
            data = [Document(page_content=json_string,metadata={"source": "local"})]
    else:
        print('Document format is not supported')
        return None
    # data = loader.load()
    return data

def chunk_data(data,chunk_size=256,chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = BedrockEmbeddings(model_id= "amazon.titan-embed-text-v1", credentials_profile_name='default')
    # vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, query,k=3):
    from langchain.chains import RetrievalQA

    llm = Bedrock(
        credentials_profile_name='default',
        model_id='amazon.titan-text-express-v1',
        model_kwargs={
            "temperature": 0.1
        }
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(query)
    return answer

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:06f}')
    return total_tokens,total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def flush_data():
    st.session_state.user_question = ''
    st.session_state.llm_user = ''
    st.session_state.history = ''
    with open('chat_history.txt', 'w') as chat_file:
        chat_file.write('')

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # st.set_page_config(layout="wide")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://s3.wns.com/S3_5/Images/Capabilities/Analytics/triange-generative-ai/Gen-AI-powered.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title(':orange[Gen AI Content Creator]')

    with st.sidebar:

        uploaded_file = st.file_uploader('**Upload a file**:', type=['.pdf', '.docx', 'txt', '.csv', '.xlsx', '.yaml'])
        content_api_key = st.text_input('**API Key:**', key='content_api_key')
        chunk_size = st.number_input('**Chunk size**:', min_value=100, max_value=2048, value=512,
                                     on_change=clear_history)
        k = st.number_input('**k**:', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Submit', on_click=clear_history)
        flush_data = st.button('Clear Form', on_click=flush_data)

        if uploaded_file and add_data:
            with st.spinner('Uploading file and chunking...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost:${embedding_cost:.4f}')
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success(':orange[File uploaded, chunked and embedded successfully.]')

    app_option = st.selectbox('**:orange[Action Menu]**', ('Please select the action',
                                                           f'Generate content for {content_api_key} API:Introduction',
                                                           f'Generate content for {content_api_key} API:Methodology',
                                                           f'Generate content for {content_api_key} API:Sample Request Response',
                                                           'Q&A App'))

    api_content_data = {}

    if 'Introduction' in app_option:
        vector_store = st.session_state.vs
        intro_answer = ask_and_get_answer(vector_store, 'Generate a brief introduction of the API in two paragraphs', k)
        if intro_answer:
            st.text_area('**:orange[Content:]** ', value=intro_answer, height=500, key='intro_content')
            save_intro = st.button('Save')
            if save_intro:
                api_content_data['introduction'] = intro_answer
    if 'Methodology' in app_option:
        vector_store = st.session_state.vs
        methodology_answer = ask_and_get_answer(vector_store, 'Generate Methodology of the API in two paragraphs', k)
        if methodology_answer:
            st.text_area('**:orange[Content:]** ', value=methodology_answer, height=500, key='methodology_content')
            save_methodology = st.button('Save')
            if save_methodology:
                api_content_data['methodology'] = methodology_answer
    if 'Sample Request Response' in app_option:
        vector_store = st.session_state.vs
        req_res_answer = ask_and_get_answer(vector_store, 'Generate Sample Request Response of the API with details', k)
        if req_res_answer:
            st.text_area('**:orange[Content:]** ', value=req_res_answer, height=500, key='req_res_content')
            save_req_res = st.button('Save')
            if save_req_res:
                api_content_data['req_res'] = req_res_answer
    if api_content_data:
        # content_doc = Document()
        content_pdf = fpdf.FPDF()

        if 'introduction' in api_content_data:
            # content_doc.add_heading('Introduction', 0)
            # p = content_doc.add_paragraph(api_content_data['introduction'])
            content_pdf.add_page()
            content_pdf.set_font("Arial", size=16, style="B")
            content_pdf.cell(200, 18, txt="Introduction", ln=True, align='L')
            content_pdf.set_font("Arial", size=12)
            content_pdf.multi_cell(180, 8, txt=api_content_data['introduction'])
            st.success(':orange[Introduction added successfully.]')
        if 'methodology' in api_content_data:
            # content_doc.add_heading('Methodology', 0)
            # p = content_doc.add_paragraph(api_content_data['methodology'])
            content_pdf.add_page()
            content_pdf.set_font("Arial", size=16, style="B")
            content_pdf.cell(200, 18, txt="Methodology", ln=True, align='L')
            content_pdf.set_font("Arial", size=12)
            content_pdf.multi_cell(180, 8, txt=api_content_data['methodology'])
            st.success(':orange[Methodology added successfully.]')
        if 'req_res' in api_content_data:
            # content_doc.add_heading('Sample Request Response', 0)
            # p = content_doc.add_paragraph(api_content_data['req_res'])
            content_pdf.add_page()
            content_pdf.set_font("Arial", size=16, style="B")
            content_pdf.cell(200, 18, txt="Sample Request Response", ln=True, align='L')
            content_pdf.set_font("Arial", size=12)
            content_pdf.multi_cell(180, 8, txt=api_content_data['req_res'])
            st.success(':orange[Sample Request Response added successfully.]')
        # content_doc.save('pages/api_data.docx')
        content_pdf.output('api_data.pdf')
        with open('api_data.pdf', "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = F'<iframe src="data:application/pdf;base64,\
                {base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    if (app_option == 'Q&A App'):
        question = st.text_input('**:orange[Ask Question:]**', key='user_question')
        if question:
            if 'vs' in st.session_state:
                vector_store = st.session_state.vs
                answer = ask_and_get_answer(vector_store, question, k)
                if answer:
                    st.text_area('**:orange[LLM Answer:]** ', value=answer, height=500, key='llm_user')

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {question}\nA: {answer}'
            current_chat_with_history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            st.session_state.history = current_chat_with_history
            with open('chat_history.txt', 'a') as chat_file:
                chat_file.write(current_chat_with_history)
            history_value = st.session_state.history
            st.text_area(label='**:orange[Chat History]**', value=history_value, key='history', height=500)
