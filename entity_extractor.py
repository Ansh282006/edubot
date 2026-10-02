from rapidfuzz import process, fuzz
import pandas as pd

class EntityExtractor:
    def __init__(self, kb_path='data/knowledge_base.csv'):
        self.kb = pd.read_csv(kb_path)
        self.entity_names = [e.replace('_', ' ').lower() for e in self.kb['entity'].tolist()]

    def extract(self, text, threshold=80):
        text_lower = text.lower()
        match, score, _ = process.extractOne(
            text_lower,
            self.entity_names,
            scorer=fuzz.partial_ratio
        )
        if score >= threshold:
            index = self.entity_names.index(match)
            return self.kb['entity'].iloc[index]
        return None

    def get_definition(self, entity):
        row = self.kb[self.kb['entity'] == entity]
        return row['definition'].values[0] if not row.empty else None

    def get_formula(self, entity):
        row = self.kb[self.kb['entity'] == entity]
        return row['formula'].values[0] if not row.empty else None
