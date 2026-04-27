import fitz # pymupdf
import re
import time


class PDFStructurer:

    def __init__(self, pdf_path, status_callback=None):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

        self.stream = []
        self.sections = []
        self.final_chunks = []
        self.final_output = {}

        self.status = status_callback if status_callback else lambda msg: None

    # ==========================================================
    # LOG FILE
    # ==========================================================
    def log(self, message):
        """Safe status emitter (future-proof)"""
        try:
            self.status(message)
        except Exception:
            pass

    # =========================================================
    # LEVEL DETECTION
    # =========================================================
    def detect_level(self, text, font_size):

        text = text.strip()
        score = 0

        if font_size >= 16:
            score += 3
        elif font_size >= 13:
            score += 2

        if 1 <= len(text.split()) <= 10:
            score += 2

        if text.isupper():
            score += 2

        if re.fullmatch(r"\d+", text):
            return "BODY"

        if len(text.split()) > 15:
            score -= 3

        if score >= 4:
            return "H1"
        elif score >= 2:
            return "H2"
        else:
            return "BODY"

    # =========================================================
    # STEP 1: EXTRACT STREAM
    # =========================================================
    def extract_stream(self):

        total_pages = len(self.doc)
        start_time = time.time()

        self.log({
            "stage": "extract",
            "status": "start",
            "progress": 0.0,
            "message": "Starting extraction",
            "processed": 0,
            "total": total_pages,
            "speed": 0,
            "eta": None
        })

        for page_num, page in enumerate(self.doc, start=1):

            blocks = page.get_text("dict")["blocks"]

            for b in blocks:
                if "lines" not in b or not b["lines"]:
                    continue

                text = ""
                sizes = []

                for line in b["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                        sizes.append(span["size"])

                text = text.strip()

                if text:
                    self.stream.append({
                        "page": page_num,
                        "text": text,
                        "font_size": max(sizes) if sizes else 0
                    })

            # ================================
            # METRICS CALCULATION
            # ================================
            progress, speed, eta = self.compute_metrics(
                start_time,
                page_num,
                total_pages
            )

            # ================================
            # EMIT EVENT (CONTROLLED)
            # ================================
            self.log({
                "stage": "extract",
                "status": "running",
                "progress": round(progress, 3),
                "message": f"Processing page {page_num}/{total_pages}",
                "processed": page_num,
                "total": total_pages,
                "speed": round(speed, 2),
                "eta": round(eta, 2)
            })

        # ================================
        # DONE EVENT
        # ================================
        self.log({
            "stage": "extract",
            "status": "done",
            "progress": 1.0,
            "message": f"Extraction complete ({len(self.stream)} blocks)",
            "processed": total_pages,
            "total": total_pages,
            "speed": round(speed, 2),
            "eta": 0
        })

    # =========================================================
    # STEP 2: CLASSIFY
    # =========================================================
    def classify(self):

        total = len(self.stream)
        start_time = time.time()

        self.log({
            "stage": "structure",
            "status": "start",
            "progress": 0.0,
            "message": "Classifying headings"
        })

        for i, s in enumerate(self.stream, start=1):

            s["level"] = self.detect_level(s["text"], s["font_size"])

            if i % 100 == 0 or i == total:

                progress, speed, eta = self.compute_metrics(
                    start_time,
                    i,
                    total
                )

                self.log({
                    "stage": "structure",
                    "status": "running",
                    "progress": round(progress, 3),
                    "message": f"Processed {i}/{total}",
                    "completed": i,
                    "total": total,
                    "speed": round(speed, 2),
                    "eta": round(eta, 2)
                })

        self.log({
            "stage": "structure",
            "status": "done",
            "progress": 1.0,
            "message": "Classification complete"
        })

    # =========================================================
    # STEP 3: BUILD STRUCTURE
    # =========================================================
    def build_structure(self):

        self.log({
                "stage": "structure",
                "progress": 0.7,
                "message": "Building document hierarchy..."
            })

        self.sections = []

        current_title = {
            "title": "DOCUMENT",
            "subsections": [],
            "pages": set()
        }

        self.sections.append(current_title)
        current_subtitle = None

        for s in self.stream:

            text = s["text"]
            level = s["level"]
            page = s["page"]

            # ---------------- H1 ----------------
            if level == "H1":

                current_title = {
                    "title": text,
                    "subsections": [],
                    "pages": set()
                }

                self.sections.append(current_title)
                current_subtitle = None
                continue

            # ---------------- H2 ----------------
            elif level == "H2":

                current_subtitle = {
                    "subtitle": text,
                    "content": "",
                    "pages": set()
                }

                current_title["subsections"].append(current_subtitle)
                continue

            # ---------------- BODY ----------------
            else:

                if current_subtitle is None:

                    current_subtitle = {
                        "subtitle": "CONTINUOUS",
                        "content": "",
                        "pages": set()
                    }

                    current_title["subsections"].append(current_subtitle)

                current_subtitle["content"] += " " + text
                current_subtitle["pages"].add(page)

    # =========================================================
    # STEP 4: FINAL CLEANING
    # =========================================================
    def finalize_chunks(self):

        self.log({
                "stage": "finalize",
                "progress": 0.9,
                "message": "Creating RAG chunks..."
            })

        chunk_id = 0

        for sec in self.sections:

            for sub in sec["subsections"]:

                content = sub["content"].strip()

                if len(content.split()) == 0:
                    continue

                self.final_chunks.append({
                    "chunk_id": chunk_id,
                    "title": sec["title"],
                    "subtitle": sub["subtitle"],
                    "content": content,
                    "pages": sorted(list(sub["pages"]))
                })

                chunk_id += 1

    # =========================================================
    # STEP 5: EXPORT FORMAT (RAG READY)
    # =========================================================
    def export(self):

        self.final_output = {
            "document": getattr(self.doc, "name", "unknown"),
            "chunks": []
        }

        for c in self.final_chunks:

            self.final_output["chunks"].append({
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "subtitle": c["subtitle"],
                "content": c["content"],
                "pages": c["pages"]
            })

        return self.final_output

    # =========================================================
    # FULL PIPELINE RUN
    # =========================================================
    def run(self):

        self.log({
                "stage": "start",
                "progress": 0.0,
                "message": "Pipeline started"
            })

        self.extract_stream()
        self.classify()
        self.build_structure()
        self.finalize_chunks()

        self.log({
                    "stage": "extract_complete",
                    "status": "done",
                    "progress": 1.0,
                    "processed": len(self.final_chunks),
                    "total": len(self.final_chunks),
                    "speed": 0,
                    "eta": 0,
                    "message": f"Done! {len(self.final_chunks)} chunks created"
                }) 

        return self.export()
    

    def compute_metrics(self, start_time, completed, total):

        elapsed = time.time() - start_time

        speed = completed / elapsed if elapsed > 0 else 0

        remaining = total - completed

        eta = remaining / speed if speed > 0 else 0

        progress = completed / total if total > 0 else 0

        return progress, speed, eta