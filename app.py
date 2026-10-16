import os
import shutil
import subprocess
import gradio as gr
from collections import defaultdict

# Modern Chain Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Core Utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Gemini & Google AI Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ================= CONFIG =================
PDF_PATH = './knowledge_base/HARDWARE_TROUBLESHOOTING_GUIDE.pdf'  # your single PDF
PERSIST_DIR = './docs/chroma/'
GOOGLE_API_KEY = "AIzaSyBoRhiBq664OmHcZXXLden_uFBJP-bhnTY"
# =========================================

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


class DocumentProcessor:
    def __init__(self):
        self.persist_directory = PERSIST_DIR
        self.issue_counter = defaultdict(int)
        self.current_issue = None

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
        )

        # Load and embed the PDF automatically
        self.load_documents([PDF_PATH])
        self.split_documents()
        self.delete_embeddings()
        self.create_embeddings()

    # ---------------- Load PDF ----------------
    def load_documents(self, file_paths):
        self.docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            self.docs.extend(loader.load())
        print('Document Loaded Successfully!')

    # ---------------- Split ----------------
    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        self.splits = text_splitter.split_documents(self.docs)

    # ---------------- Delete old embeddings ----------------
    def delete_embeddings(self):
        if os.path.isdir(self.persist_directory):
            print('Cleaning up old database...')
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    # ---------------- Create embeddings ----------------
    def create_embeddings(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectordb_doc = Chroma.from_documents(
            documents=self.splits,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )

    # ---------------- Get embeddings ----------------
    def get_embeddings(self):
        return self.vectordb_doc

    # ---------------- Document chain ----------------
    def document_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        ### SYSTEM ROLE
        You are a professional electronics troubleshooting assistant.
        You provide accurate, safe, and structured troubleshooting guidance based strictly on the supplied Context.

        You must not assume, infer, or invent information beyond the provided Context.

        ---

        ### RESPONSE STYLE (MANDATORY)
        - Use a calm, professional, human-like tone similar to ChatGPT or Gemini.
        - Guide the user naturally through the troubleshooting process.
        - Use short transitional phrases such as:
        “Let's start by…”
        “This usually indicates…”
        “Next, we'll check…”
        - Explanations must feel conversational but remain technically precise.
        - Avoid sounding like a checklist or repair manual.

        ---

        ### RESPONSE STRUCTURE (REQUIRED, FLEXIBLE FORMAT)
        Each response must clearly and completely address the following, but **without rigid section dumping**:

        1. Clearly describe the reported symptom in natural language.
        2. Explain the relevant possible causes (only from Context).
        3. Walk the user through a logical, step-by-step diagnostic flow:
        - One action at a time
        - Explain what the step checks and why it matters
        4. Provide corrective actions tied directly to the diagnostic results.

        Section headings are optional.
        Clarity and flow are more important than labels.

        ---

        ### STEP-BY-STEP REQUIREMENTS
        - Steps must be sequential and actionable.
        - Avoid vague instructions such as “check the device”.
        - Every step must state:
        - What to do
        - What the user should observe
        - What to do next based on the outcome (if applicable)

        ---

        ### SAFETY & SCOPE RULES
        - Use ONLY information present in the provided Context.
        - Do NOT introduce new causes, tools, or repair methods.
        - Clearly indicate when a step is user-safe.
        - If a procedure involves internal components, explicitly state safety precautions if present in Context.

        ---
                                                  
        ### PARTIAL CONTEXT HANDLING
        If a diagnostic step is identified but full procedural steps are not present in the retrieved Context:
        - Explain what the step is meant to check
        - Clearly state that detailed instructions are not available in the current context
        - Ask the user if they want guidance on that specific step

        Do NOT escalate the issue at this stage.
        
        ---

        ### ISSUE ESCALATION LOGIC
            #### ESCALATION RULE (STRICT)
                Do NOT include the issue escalation notice unless the user explicitly states that:
                - the problem is still not fixed
                - or the suggested steps did not work
                - or the issue persists after following the guidance

                If the user has NOT said this explicitly, you MUST NOT include the escalation notice.

                                                  
        ⚠️ ISSUE ESCALATION NOTICE

        This problem appears to be persistent and may require professional servicing.
        Please contact the authorized service center:
        📧 example@gmail.com
        📞 +1-555-555-0123                                      

        Do not attempt further repairs if the device is under warranty.

        ---

        ### FORBIDDEN BEHAVIOR
        - No guessing or speculation.
        - No casual language, jokes, or emotional expressions.
        - No information outside the provided Context.
        - No skipping diagnostic reasoning.

        ### Context
        ---
        {context}
        ---

        ### User Question
        {input}

        ### Response:
        """)
        return create_stuff_documents_chain(self.llm, prompt)

    # ---------------- Retrieval chain ----------------
    def reterival_chain(self, document_chain, document_embeddings):
        retriever = document_embeddings.as_retriever(search_kwargs={"k": 5})
        return create_retrieval_chain(retriever, document_chain)

     # ---------------- Main response ----------------
    def get_response(self, user_message):
        document_chain = self.document_chain()
        retrieval_chain = self.reterival_chain(document_chain, self.get_embeddings())
        response = retrieval_chain.invoke({"input": user_message})
        return response["answer"]


# ================= Gradio App =================
def main():
    processor = DocumentProcessor()

    def chat_fn(message, history):
        return processor.get_response(message)

    ACCENT_COLOR = "#1E90FF"  # Dodger Blue, can be changed

    with gr.Blocks(theme=gr.themes.Soft()) as demo:

        # ================= HEADER =================
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:30px; background-color:#f5f5f5; border-radius:10px;">
                <h1 style="color:{ACCENT_COLOR}; margin-bottom:5px;">🔧 The Hardware Whisperer</h1>
                <p style="font-size:16px; color:#333; margin-top:0;">
                    Step-by-step diagnostics powered by official service manuals
                </p>
            </div>
            """
        )

        # ================= SUPPORTED DEVICES =================
        gr.Markdown("### ✅ Supported Devices", elem_classes="section-title")

        with gr.Row(equal_height=True):
            for icon, name in [("📱", "Mobile Phones"), ("💻", "Laptops"), 
                               ("📺", "Televisions"), ("🌀", "Fans"), ("❄️", "Air Conditioners")]:
                gr.Markdown(
                    f"""
                    <div style="text-align:center; padding:15px; border-radius:10px;
                                border: 1px solid #ddd; background-color:#fafafa;
                                box-shadow:0 2px 5px rgba(0,0,0,0.05);">
                        <div style="font-size:30px; color:{ACCENT_COLOR};">{icon}</div>
                        <div style="font-weight:600; margin-top:5px; color:#333;">{name}</div>
                    </div>
                    """
                )

        # ================= INFO =================
        gr.Markdown(
            """
            <hr style="margin:30px 0;">
            <p style="font-size:15px; color:#444;">
            This assistant helps diagnose <b>hardware-related issues</b> such as power failure,
            display problems, overheating, abnormal noise, and non-responsive devices.
            </p>
            """
        )

        # ================= HOW IT WORKS =================
        gr.Markdown("### 🛠️ How to Use", elem_classes="section-title")

        gr.Markdown(
            f"""
            <ol style="font-size:15px; line-height:1.8; color:#444;">
                <li>Describe the problem clearly  
                    <i>(Example: "Laptop turns on but screen is black")</i></li>
                <li>The assistant will analyze official manuals</li>
                <li>You will receive a professional step-by-step guide</li>
            </ol>
            """
        )

        # ================= CHAT =================
        gr.Markdown("### 💬 Start Troubleshooting", elem_classes="section-title")

        gr.ChatInterface(
            fn=chat_fn,
            textbox=gr.Textbox(
                placeholder="Describe your hardware problem in detail...",
                lines=1
            ),
            examples=[
                "My phone is not turning on",
                "Laptop overheats and shuts down",
                "AC turns on but does not cool"
            ],
        )

        # ================= FOOTER =================
        gr.Markdown(
            """
            <hr style="margin:30px 0;">
            <p style="text-align:center; font-size:12px; color:#888;">
                ⚠️ Follow safety instructions. Do not attempt repairs under warranty.
            </p>
            """
        )

    demo.launch()


if __name__ == "__main__":
    main()
