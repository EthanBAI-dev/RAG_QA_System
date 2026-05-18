#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def _get_secret(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


class BaseEmbeddings:
    """
    埋め込みモデルの基底クラス
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        入力テキストをベクトル（embedding）に変換します。
        引数:
            text: 入力テキスト文字列
            model: 使用するモデル名（openai / zhipu など）
        戻り値:
            テキストの意味を表すベクトル（List[float]）
        注意:
            ここでは具体的なロジックは実装されておらず、サブクラスで必ずオーバーライドしてください。
        """
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        2つのベクトル間のコサイン類似度を計算します。
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    OpenAI 埋め込みモデル
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = _get_secret("OPENAI_API_KEY")
            self.client.base_url = _get_secret("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError

class JinaEmbedding(BaseEmbeddings):
    """
    Jina 埋め込みモデル
    """
    def __init__(self, path: str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()
        
    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()
    
    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model

class ZhipuEmbedding(BaseEmbeddings):
    """
    ZhipuAI 埋め込みモデル
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=_get_secret("ZHIPUAI_API_KEY")) 
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-2",
        input=text,
        )
        return response.data[0].embedding