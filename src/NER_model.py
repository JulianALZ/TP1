import spacy
import pandas as pd
import re
from abc import ABC, abstractmethod
from typing import List
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError as e:
    logger.error("Le modèle spaCy français n'est pas installé. Veuillez le télécharger avant d'exécuter ce script.")
    raise e

class TextProcessor(ABC):
    """
    Classe abstraite définissant une interface pour les processeurs de texte.
    """

    @abstractmethod
    def clean_text(self, text: str) -> str:
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        pass

class NER_Model(TextProcessor):
    """
    Processeur de texte qui utilise spaCy pour l'extraction d'entités nommées.
    """

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)  # Supprime la ponctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Supprime les espaces excédentaires
        return text

    def extract_entities(self, text: str) -> List[str]:
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PER"]

    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'video_name') -> pd.DataFrame:
        if text_column not in df.columns:
            logger.error(f"La colonne '{text_column}' n'est pas présente dans le dataframe.")
            raise ValueError(f"La colonne '{text_column}' est manquante dans le dataframe.")

        df[text_column] = df[text_column].apply(self.clean_text)
        df['NER_result'] = df[text_column].apply(self.extract_entities)
        return df

    def find_videos_with_name(self, df: pd.DataFrame, name: str, text_column: str = 'video_name') -> pd.DataFrame:
        """
        Filtre le dataframe pour retourner les lignes où le nom donné apparaît dans les titres de vidéo.

        :param df: dataframe Pandas à filtrer.
        :param name: nom à rechercher dans les titres de vidéos.
        :param text_column: nom de la colonne contenant les titres de vidéos.
        :return: dataframe Pandas contenant uniquement les lignes avec les titres de vidéos où le nom apparaît.
        """
        # Vérification que la colonne existe dans le dataframe
        if text_column not in df.columns:
            raise ValueError(f"La colonne '{text_column}' est manquante dans le dataframe.")

        # Filtrage des vidéos qui contiennent le nom spécifié
        filtered_df = df[df[text_column].str.contains(name, case=False, na=False)]
        return filtered_df
