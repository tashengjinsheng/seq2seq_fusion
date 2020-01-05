"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.knowledge_model import KnowledgeModel

__all__ = ["build_model_saver", "ModelSaver", "NMTModel", "KnowledgeModel"]
