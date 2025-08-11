# VJ 2025/08/10 - Web/output-only variant
# FAQ Generator based on Text
# This version prints FAQs to the console instead of saving .txt / .jsonl files.

import json
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
import PyPDF2
from textstat.textstat import textstat

class AssignmentFAQGenerator:
    def __init__(self):
        # Download required NLTK data if missing
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

        # Stop words
        self.stop_words = set(stopwords.words('english'))

        # Internal storage
        self.raw_text = ""
        self.sentences: List[str] = []
        self.key_concepts: List[str] = []
        self.faq_pairs: List[Dict[str, str]] = []

    def extract_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for p in range(len(reader.pages)):
                    page = reader.pages[p]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"[ERROR] reading PDF: {e}")
            return ""
        return text

    def extract_from_txt(self, txt_path: str) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[ERROR] reading TXT: {e}")
            return ""

    def preprocess_text(self, text: str) -> str:
        # Normalize whitespace and newlines
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)        # collapse multiple spaces
        text = re.sub(r'\n+', '\n', text)       # collapse multiple newlines
        # Remove unusual characters but keep punctuation useful for sentences
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        # Remove common pdf page markers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+/\d+', '', text)
        return text.strip()

    def extract_key_concepts(self, text: str, top_n: int = 20) -> List[str]:
        words = word_tokenize(text.lower())
        pos_tags = pos_tag(words)
        important = []
        for word, pos in pos_tags:
            if (pos.startswith(('NN', 'JJ', 'VB')) and
                word not in self.stop_words and
                len(word) > 2 and
                word.isalpha()):
                important.append(word)
        freq = Counter(important)
        return [w for w, _ in freq.most_common(top_n)]

    def generate_definition_questions(self, concepts: List[str], text: str) -> List[Dict[str, str]]:
        questions = []
        sentences = sent_tokenize(text)
        for concept in concepts:
            q = f"What is {concept}?"
            concept_lower = concept.lower()
            definition_sentences = []
            for sent in sentences:
                s_low = sent.lower()
                if concept_lower in s_low and any(ind in s_low for ind in ['is', 'are', 'defined as', 'refers to', 'means', 'definition', 'is called', 'is known as']):
                    definition_sentences.append(sent.strip())
            if definition_sentences:
                a = max(definition_sentences, key=len)
                questions.append({"question": q, "answer": a})
        return questions

    def generate_how_questions(self, text: str) -> List[Dict[str, str]]:
        questions = []
        sentences = sent_tokenize(text)
        process_indicators = ['first', 'then', 'next', 'finally', 'step', 'process', 'method', 'procedure', 'how to', 'in order to', 'steps']
        process_sentences = [s for s in sentences if any(ind in s.lower() for ind in process_indicators)]
        for sentence in process_sentences[:5]:
            words = word_tokenize(sentence.lower())
            pos_tags = pos_tag(words)
            main_verbs = [w for w, p in pos_tags if p.startswith('VB')]
            if main_verbs:
                main_verb = main_verbs[0]
                q = f"How do you {main_verb} in this context?"
                questions.append({"question": q, "answer": sentence.strip()})
        return questions

    def generate_why_questions(self, text: str) -> List[Dict[str, str]]:
        questions = []
        sentences = sent_tokenize(text)
        reason_indicators = ['because', 'due to', 'since', 'as a result', 'therefore', 'consequently', 'reason', 'cause', 'leads to', 'results in']
        for sent in sentences:
            s_low = sent.lower()
            if any(ind in s_low for ind in reason_indicators):
                q = "Why is this concept important in the assignment?"
                questions.append({"question": q, "answer": sent.strip()})
        return questions[:3]

    def generate_comparison_questions(self, text: str, concepts: List[str]) -> List[Dict[str, str]]:
        questions = []
        sentences = sent_tokenize(text)
        comparison_words = ['versus', 'vs', 'compared to', 'difference between', 'similar to', 'unlike', 'whereas', 'however', 'but']
        comparison_sentences = [s for s in sentences if any(w in s.lower() for w in comparison_words)]
        for i in range(min(len(concepts)-1, 3)):
            c1 = concepts[i]
            c2 = concepts[i+1]
            q = f"What is the difference between {c1} and {c2}?"
            relevant = [s for s in comparison_sentences if c1.lower() in s.lower() and c2.lower() in s.lower()]
            if relevant:
                a = relevant[0].strip()
            else:
                a = f"Both {c1} and {c2} are important concepts in this assignment that serve different roles in the overall context."
            questions.append({"question": q, "answer": a})
        return questions

    def remove_duplicate_questions(self, faq_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unique = []
        seen = set()
        for p in faq_pairs:
            key = p['question'].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def process_document(self, input_str: str) -> bool:
        # Reset internal state for repeated runs
        self.raw_text = ""
        self.sentences = []
        self.key_concepts = []
        self.faq_pairs = []

        # If input_str looks like a file path with .pdf or .txt, read file; otherwise treat as raw text
        input_lower = input_str.lower().strip()
        if input_lower.endswith('.pdf'):
            self.raw_text = self.extract_from_pdf(input_str)
        elif input_lower.endswith('.txt'):
            self.raw_text = self.extract_from_txt(input_str)
        else:
            # Treat as raw text pasted by user
            self.raw_text = input_str

        if not self.raw_text:
            print("[ERROR] No text found or could not extract text.")
            return False

        processed = self.preprocess_text(self.raw_text)
        self.sentences = sent_tokenize(processed)
        self.key_concepts = self.extract_key_concepts(processed, top_n=15)

        definition_qs = self.generate_definition_questions(self.key_concepts, processed)
        how_qs = self.generate_how_questions(processed)
        why_qs = self.generate_why_questions(processed)
        comparison_qs = self.generate_comparison_questions(processed, self.key_concepts)

        self.faq_pairs = definition_qs + how_qs + why_qs + comparison_qs
        self.faq_pairs = self.remove_duplicate_questions(self.faq_pairs)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        if not self.raw_text:
            return {"error": "No document processed yet."}
        reading = None
        try:
            reading = textstat.flesch_reading_ease(self.raw_text)
        except Exception:
            reading = "N/A"
        avg_sentence_len = (len(self.raw_text.split()) / len(self.sentences)) if self.sentences else 0
        return {
            "total_characters": len(self.raw_text),
            "total_sentences": len(self.sentences),
            "total_faqs": len(self.faq_pairs),
            "key_concepts_found": len(self.key_concepts),
            "key_concepts": self.key_concepts[:10],
            "reading_level": reading,
            "avg_sentence_length": avg_sentence_len
        }


def main():
    print("=== Assignment FAQ Generator (console output only) ===")
    print("You may either enter a local file path ending with .pdf or .txt, OR paste the assignment text directly.")
    print("If you choose to paste text, just paste it and press Enter when done (single-line inputs also work).")
    user_input = input("\nEnter file path OR paste text here: ").strip()

    faq_generator = AssignmentFAQGenerator()
    success = faq_generator.process_document(user_input)

    if not success:
        print("Processing failed. Please check the input.")
        return

    # Print statistics
    stats = faq_generator.get_statistics()
    print("\n--- Document Statistics ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Pretty-print Q&A pairs
    print("\n--- Generated FAQs ---")
    if not faq_generator.faq_pairs:
        print("No FAQ pairs generated.")
    else:
        for i, pair in enumerate(faq_generator.faq_pairs, start=1):
            print(f"Q{i}: {pair['question']}")
            print(f"A{i}: {pair['answer']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
