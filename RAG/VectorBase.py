#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
モジュール名: 軽量ローカルベクトルデータベース (Vector Store)
機能説明:
    本モジュールは RAG（検索拡張生成）システム向けの基礎的なベクトルデータベースを実装します。
    LLM の「外部記憶」として機能し、以下の主要ワークフローを担当します：

    1. ベクトル化 (Embedding): 分割されたテキストチャンクを受け取り、指定された Embedding モデルを用いて高次元ベクトルに変換します。
    2. 永続化 (Persistence): 計算済みの高次元ベクトルと原文テキストを JSON 形式でローカルディスクに保存し、再計算のオーバーヘッドを回避します。
    3. 類似度検索 (Query): ユーザーの質問をベクトル化し、「コサイン類似度」によりデータベース内の全ベクトルと比較し、最も適合する Top-K チャンクを返します。

依存関係:
    - `RAG.Embeddings` のベースモデルインターフェースに依存します。
    - `numpy` を用いて高速な行列演算とソートを行います。
    - `tqdm` を用いて長文処理時のプログレスバーを表示します。
=============================================================================
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import json
from RAG.Embeddings import BaseEmbeddings, OpenAIEmbedding, JinaEmbedding, ZhipuEmbedding
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: List[str] = [''], sources: List[str] = None) -> None:
        """
        ベクトルデータベースを初期化します。
        :param document: 処理対象のドキュメントリスト（文字列配列）。
        :param sources: 各チャンクの出典ファイル名（document と同じ長さ）。
        """
        self.document = document
        self.sources = sources if sources else [''] * len(document)

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """
        全ドキュメントを走査し、指定された埋め込みモデルでベクトルを生成します。
        """
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        """
        ドキュメントデータと対応するベクトルデータをローカル JSON ファイルに永続化します。
        """
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)

        with open(f"{path}/sources.json", 'w', encoding='utf-8') as f:
            json.dump(self.sources, f, ensure_ascii=False)

        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        vectors_path = f"{path}/vectors.json"
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"ベクトルファイルが見つかりません: {vectors_path}\n"
                f"最初にデータをベクトル化して保存してください。"
            )

        with open(vectors_path, 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)

        doc_path = f"{path}/documents.json"
        if not os.path.exists(doc_path):
            raise FileNotFoundError(
                f"文書ファイルが見つかりません: {doc_path}\n"
                f"最初にデータをベクトル化して保存してください。"
            )

        with open(doc_path, 'r', encoding='utf-8') as f:
            self.document = json.load(f)

        sources_path = f"{path}/sources.json"
        if os.path.exists(sources_path):
            with open(sources_path, 'r', encoding='utf-8') as f:
                self.sources = json.load(f)
        else:
            self.sources = [''] * len(self.document)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        2つのベクトル間のコサイン類似度を計算します。1 に近いほどテキストの意味が類似しています。
        """
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[Dict]:
        """
        ユーザーの質問に最も関連するドキュメントを検索します。
        :param query: ユーザーの質問文字列。
        :param EmbeddingModel: 質問をベクトル化する埋め込みモデル。
        :param k: 返却する最関連ドキュメントの数 (Top-K)。
        :return: List[Dict] 各要素に text, score, source を含む
        """
        query_vector = EmbeddingModel.get_embedding(query)

        scores = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])

        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.document[idx],
                "score": float(scores[idx]),
                "source": self.sources[idx] if idx < len(self.sources) else "",
            })
        return results

    def remove_by_source(self, source_to_remove: str, persist_path: str = 'storage') -> Tuple[int, int]:
        new_docs, new_sources, new_vectors = [], [], []
        removed = 0
        kept = 0
        for i, src in enumerate(self.sources):
            if src == source_to_remove:
                removed += 1
            else:
                new_docs.append(self.document[i])
                new_sources.append(src)
                if hasattr(self, 'vectors') and i < len(self.vectors):
                    new_vectors.append(self.vectors[i])
                kept += 1
        self.document = new_docs
        self.sources = new_sources
        self.vectors = new_vectors
        if persist_path:
            self.persist(persist_path)
        return removed, kept

    def append(self, new_docs: List[str], new_sources: List[str],
               new_vectors: List[List[float]] = None, persist_path: str = 'storage'):
        self.document.extend(new_docs)
        self.sources.extend(new_sources)
        if new_vectors is not None:
            if not hasattr(self, 'vectors'):
                self.vectors = []
            self.vectors.extend(new_vectors)
        if persist_path:
            self.persist(persist_path)