import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from ollama import chat
import tiktoken
from google import genai


class LocalRAGPipeline:

    def __init__(
        self,
        api_key, 
        embed_model_path="./models/all-MiniLM-L6-v2",
        reranker_path= "./models/cross-encoder/ms-marco-MiniLM-L-6-v2",
        db_path="./database/chroma_db",
        collection_name="pdf_rag",
        max_tokens=7000,
        doc_id=None,
        mode="single",
    ):

        # =====================================================
        # MODELS
        # =====================================================
        self.embed_model = SentenceTransformer(embed_model_path)
        self.reranker = CrossEncoder(reranker_path)

        self.doc_id = doc_id
        self.mode = mode

        # =====================================================
        # CHROMA DB
        # =====================================================
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        self.clients = genai.Client(api_key=api_key)

        # =====================================================
        # TOKENIZER
        # =====================================================
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

    # =========================================================
    # TOKEN CONTROLLER
    # =========================================================
    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def build_context(self, blocks):

        context = []
        total = 0

        for b in blocks:

            tokens = self.count_tokens(b)

            if total + tokens > self.max_tokens:
                break

            context.append(b)
            total += tokens

        return "\n\n".join(context)

    # =========================================================
    # QUERY EXPANSION
    # =========================================================
    def expand_query(self, query, model="llama3.2"):

        #prompt = f"""
        #            Generate 5 semantic search queries for document retrieval.

        #            Rules:
        #            - Focus on likely document terminology
        #            - Use domain keywords
        #            - Preserve original intent
        #            - Avoid generic phrases
        #            - Make each query meaningfully different
        #            - No numbering
        #            - One per line
        #            """
        #query_to_expand = f"""
        #                        Question:
        #                        {query}
        #                        """

        #response = chat(
        #    model=model,
        #    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query_to_expand}]
        #)

        #text = response["message"]["content"]
        prompt = f"""
                    Generate 10 semantic search queries for document retrieval.

                    Rules:
                    - Focus on likely document terminology
                    - Use domain keywords
                    - Preserve original intent
                    - Avoid generic phrases
                    - Make each query meaningfully different
                    - No numbering
                    - One per line
                    
                    Question:
                    {query}
                    """
        response = self.clients.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
        text = str(response.text)

        queries = [q.strip("- ").strip()
                for q in text.split("\n")
                if q.strip()]

        # include original query
        queries.insert(0, query)
        #print(list(set(queries)))

        return list(set(queries))

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================
    def semantic_search(self, query, top_k=50, all_docs=False):

        q_emb = self.embed_model.encode(query).tolist()

        if self.mode == "all":
            where = None
        else:
            where = {"doc_id": self.doc_id}

        #print(q_emb)

        results = self.collection.query( query_embeddings=[q_emb], n_results=top_k, where=where )

        #print(results)

        return results

    # =========================================================
    # FORMAT CHROMA RESULTS
    # =========================================================
    def format_blocks(self, results):

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        blocks = []

        for i in range(len(docs)):

            m = metas[i]

            blocks.append({
                            "text": docs[i],
                            "title": m.get("title", ""),
                            "subtitle": m.get("subtitle", ""),
                            "pages": m.get("pages", ""),
                            "chunk_id": m.get("chunk_id", i),
                            "subchunk_index": m.get("subchunk_index", 0)
                        })

        return blocks

    # =========================================================
    # BM25
    # =========================================================
    def build_bm25(self, blocks):

        texts = [b["text"] for b in blocks]
        tokenized = [t.split() for t in texts]

        return BM25Okapi(tokenized)

    def bm25_rank(self, query, blocks, bm25, top_k=15):

        scores = bm25.get_scores(query.split())

        ranked = sorted(
            zip(blocks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in ranked[:top_k]]

    # =========================================================
    # RERANKING
    # =========================================================
    def rerank(self, query, blocks):

        pairs = [(query, b["text"]) for b in blocks]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(blocks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in ranked]

    # =========================================================
    # CONTEXT BUILDER
    # =========================================================
    def build_formatted_context(self, blocks):

        formatted = []

        for i, b in enumerate(blocks):

            formatted.append(f"""
                    [CHUNK {i}]
                    TITLE: {b['title']}
                    SUBTITLE: {b['subtitle']}
                    PAGES: {b['pages']}
                    CHUNK_ID: {b['chunk_id']}

                    TEXT:
                    {b['text']}
                    """.strip())

        #formatted.sort(key=lambda x: len(x))
        return self.build_context(formatted)

    # =========================================================
    # LLM CALL
    # =========================================================
    def ask_llm(self, context, question, model="llama3.2"):

        #prompt = f"""
        #            You are a precise document QA system.

        #            RULES:
        #            - Use ONLY the context or previous conversation below
        #            - If answer is not found, say "Not found in document"
        #            - You must always cite chunks clearly like ( from Title: [TITLE], pg. [x])
        #            - If you are asked a question about the book you can get it from the context and answer
        #            - May sure to use markdown when necessary
        #        """
        #query_prompt = f"""
        #                    CONTEXT:
        #                    {context}

        #                    QUESTION:
        #                    {question}
        #                    """

        #response = chat(
        #    model=model,
        #    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query_prompt}]
        #)

        #return response["message"]["content"]
        prompt = f"""
                    You are a precise document QA system.

                    RULES:
                    - Use ONLY the context or previous conversation below
                    - If answer is not found, say "Not found in document"
                    - You must always cite chunks clearly like ( from Title: [TITLE], pg. [x])
                    - If you are asked a question about the book you can get it from the context and answer
                    - May sure to use markdown when necessary

                    CONTEXT:
                    {context}

                    QUESTION:
                    {question}
                    """
        response = self.clients.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
        result = str(response.text)

        return result
    
    def ask_llm_stream(self, context, question, model="llama3.2"):

        prompt = f"""
                    You are a truthful and careful AI assistant.

                    Use retrieved context as reference material, NOT as guaranteed truth.

                    Rules:
                    - Never invent facts, citations, sections, APIs, or technical details.
                    - If retrieved context is incorrect, inconsistent, or unrelated, say so clearly.
                    - Verify logical consistency before answering.
                    - Distinguish between confirmed facts and uncertain information.
                    - If unsure, say “I cannot verify this confidently.”
                    - Prefer accuracy over confidence.
                    - Correct wrong information instead of repeating it.
                    - For legal questions, be strict with section numbers and terminology.
                    - For technical questions, avoid fake libraries, commands, or functions.
                    - Keep answers clear, structured, meaningful, very well explained and broadly.
                    - If context conflicts with known facts, explain the conflict and give the corrected explanation.
                    - Make sure to always use markdown when necessary.
                """
        query_prompt = f"""
                            CONTEXT:
                            {context}

                            QUESTION:
                            {question}
                            """
        stream = chat(
            model=model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query_prompt}],
            stream=True
        )

        for chunk in stream:
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                if content:
                    yield content

    # =========================================================
    # MAIN RAG PIPELINE
    # =========================================================
    def run(self, query):

        # 1. QUERY EXPANSION
        queries = self.expand_query(query)

        all_blocks = []

        for q in queries:

            results = self.semantic_search(q, top_k=100)
            all_blocks.extend(self.format_blocks(results))

        # =====================================================
        # NO RESULTS SAFETY
        # =====================================================
        if not all_blocks:
            return "No relevant information found in this PDF."
        
        # =====================================================
        # REMOVE DUPLICATE CHUNKS
        # =====================================================
        unique_blocks = {}
            
        for block in all_blocks:

            chunk_id = block["chunk_id"]

            if chunk_id not in unique_blocks:
                unique_blocks[chunk_id] = block

        all_blocks = list(unique_blocks.values())

        # 2. BM25
        bm25 = self.build_bm25(all_blocks)
        bm25_blocks = self.bm25_rank(query, all_blocks, bm25)

        # =====================================================
        # 3. RERANKING
        # =====================================================
        reranked = self.rerank(query, bm25_blocks)

        # =====================================================
        # RESTORE DOCUMENT ORDER
        # =====================================================
        def get_chunk_position(block):

            # ---------------------------------------------
            # PAGE NUMBER
            # ---------------------------------------------
            pages = str(block["pages"])

            try:
                first_page = int(pages.split(",")[0])

            except:
                first_page = 999999

            # ---------------------------------------------
            # SUBCHUNK ORDER
            # ---------------------------------------------
            subchunk = block.get("subchunk_index", 0)

            return (first_page, subchunk)

        # =====================================================
        # KEEP MOST RELEVANT CHUNKS
        # =====================================================
        #top_chunks = reranked[:90]

        # =====================================================
        # RESTORE NATURAL DOCUMENT ORDER
        # =====================================================
        #top_chunks.sort(key=get_chunk_position)
        reranked.sort(key=get_chunk_position)

        # =====================================================
        # BUILD CONTEXT
        # =====================================================
        #context = self.build_formatted_context(top_chunks)
        context = self.build_formatted_context(reranked)

        # 5. TOKEN SAFETY CHECK
        if self.count_tokens(context) > self.max_tokens:
            context = self.build_formatted_context(reranked[:60])

        # 6. FINAL ANSWER
        return self.ask_llm(context, query)
