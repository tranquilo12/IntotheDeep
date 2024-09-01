import os
import numpy as np
import requests
import pathspec
from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dataclasses import dataclass
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from rich.progress import Progress
from rich.console import Console

console = Console()


@dataclass
class CodeChunk:
    """
    Represents a chunk of code with metadata.

    Attributes
    ----------
    content : str
        The actual code content.
    chunk_type : str
        Type of the chunk (e.g., 'function', 'class', 'file').
    start_byte : int
        Starting byte position in the original file.
    end_byte : int
        Ending byte position in the original file.
    start_point : tuple
        Starting (line, column) in the original file.
    end_point : tuple
        Ending (line, column) in the original file.
    file_path : str
        Path to the file containing this chunk.
    """

    content: str
    chunk_type: str
    start_byte: int
    end_byte: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    file_path: str


def setup_tree_sitter() -> Parser:
    """
    Set up the tree-sitter parser for Python.

    Returns
    -------
    Parser
        Configured tree-sitter parser for Python.
    """
    py_lang = Language(tspython.language())
    parser = Parser(language=py_lang)
    return parser


def initialize_qdrant() -> QdrantClient:
    """
    Initialize an in-memory Qdrant client and create a collection.

    Returns
    -------
    QdrantClient
        Configured Qdrant client with an 'IntoTheDeep' collection.
    """
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name="IntoTheDeep",
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    return client


def load_gitignore(repo_path: str) -> pathspec.PathSpec:
    """
    Load .gitignore rules from the repository.

    Parameters
    ----------
    repo_path : str
        Path to the repository root.

    Returns
    -------
    pathspec.PathSpec or None
        PathSpec object with gitignore rules, or None if .gitignore doesn't exist.
    """
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as gitignore_file:
            return pathspec.PathSpec.from_lines("gitwildmatch", gitignore_file)
    return None


def is_ignored(path: str, gitignore_spec: pathspec.PathSpec) -> bool:
    """
    Check if a path should be ignored based on .gitignore rules.

    Parameters
    ----------
    path : str
        Path to check.
    gitignore_spec : pathspec.PathSpec
        PathSpec object with gitignore rules.

    Returns
    -------
    bool
        True if the path should be ignored, False otherwise.
    """
    return gitignore_spec.match_file(path) if gitignore_spec else False


def chunk_code_file(file_path: str, parser: Parser) -> List[CodeChunk]:
    """
    Chunk a Python file into CodeChunk objects.

    Parameters
    ----------
    file_path : str
        Path to the Python file.
    parser : Parser
        Configured tree-sitter parser.

    Returns
    -------
    List[CodeChunk]
        List of CodeChunk objects representing the file content.
    """
    with open(file_path, "rb") as file:
        content = file.read()

    try:
        decoded_content = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded_content = content.decode("latin-1")
        except UnicodeDecodeError:
            console.print(f"[red]Unable to decode {file_path}. Skipping this file.")
            return []

    tree = parser.parse(bytes(decoded_content, "utf-8"))
    chunks = []

    # File-level chunk
    chunks.append(
        CodeChunk(
            decoded_content,
            "file",
            0,
            len(content),
            tree.root_node.start_point,
            tree.root_node.end_point,
            file_path,
        )
    )

    # Function-level and class-level chunks
    for node in tree.root_node.children:
        if node.type in ["function_definition", "class_definition"]:
            chunk_content = decoded_content[node.start_byte : node.end_byte]
            chunks.append(
                CodeChunk(
                    chunk_content,
                    "function" if node.type == "function_definition" else "class",
                    node.start_byte,
                    node.end_byte,
                    node.start_point,
                    node.end_point,
                    file_path,
                )
            )

    return chunks


def process_repository(
    repo_path: str, parser: Parser, progress: Optional[Progress] = None
) -> List[CodeChunk]:
    """
    Process all Python files in a repository and create CodeChunks.

    Parameters
    ----------
    repo_path : str
        Path to the repository root.
    parser : Parser
        Configured tree-sitter parser.
    progress : Optional[Progress]
        Rich Progress instance for tracking progress.

    Returns
    -------
    List[CodeChunk]
        List of CodeChunk objects from all processed Python files.
    """
    gitignore_spec = load_gitignore(repo_path)
    all_chunks = []

    # Prepare the file list
    files = [
        (root, file)
        for root, _, files in os.walk(repo_path)
        for file in files
        if file.endswith(".py")
    ]

    repo_task = (
        progress.add_task("[cyan]Processing repository...", total=len(files))
        if progress
        else None
    )

    for root, file in files:
        file_path = os.path.join(root, file)
        if not is_ignored(file_path, gitignore_spec):
            try:
                chunks = chunk_code_file(file_path, parser)
                all_chunks.extend(chunks)
            except Exception as e:
                console.print(f"[red]Error processing {file_path}: {str(e)}")
                console.print(f"[yellow]Error details: {type(e).__name__}")

        if progress:
            progress.update(repo_task, advance=1)

    return all_chunks


def get_embeddings(
    chunks: List[CodeChunk], batch_size: int = 32, progress: Optional[Progress] = None
) -> List[np.ndarray]:
    """
    Get embeddings for a list of CodeChunks using a local embedding server.

    Parameters
    ----------
    chunks : List[CodeChunk]
        List of CodeChunk objects to embed.
    batch_size : int, optional
        Number of chunks to process in each batch (default is 32).
    progress : Optional[Progress]
        Rich Progress instance for tracking progress.

    Returns
    -------
    List[np.ndarray]
        List of embedding vectors.
    """
    embeddings = []
    texts = [chunk.content for chunk in chunks]

    embed_task = (
        progress.add_task("[green]Generating embeddings...", total=len(texts))
        if progress
        else None
    )

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = requests.post(
            "http://localhost:1234/v1/embeddings",
            json={"model": "bge-large-en-v1.5-q4_k_m", "input": batch},
        )
        if response.status_code == 200:
            batch_embeddings = response.json()["data"]
            embeddings.extend([np.array(emb["embedding"]) for emb in batch_embeddings])
        else:
            raise Exception(f"Error in getting embeddings: {response.text}")

        if progress:
            progress.update(embed_task, advance=len(batch))

    return embeddings


def store_chunks(
    client: QdrantClient, chunks: List[CodeChunk], progress: Optional[Progress] = None
):
    """
    Store CodeChunks and their embeddings in Qdrant.

    Parameters
    ----------
    client : QdrantClient
        Initialized Qdrant client.
    chunks : List[CodeChunk]
        List of CodeChunk objects to store.
    progress : Optional[Progress]
        Rich Progress instance for tracking progress.
    """
    embeddings = get_embeddings(chunks, progress=progress)

    store_task = (
        progress.add_task("[blue]Storing chunks in Qdrant...", total=len(chunks))
        if progress
        else None
    )

    # Prepare points for batch insertion
    points = [
        models.PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "file_path": chunk.file_path,
                "start_byte": chunk.start_byte,
                "end_byte": chunk.end_byte,
                "start_point": chunk.start_point,
                "end_point": chunk.end_point,
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Batch insert points
    batch_size = 100  # Adjust based on your needs and Qdrant's capabilities
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name="IntoTheDeep", points=batch)
        if progress:
            progress.update(store_task, advance=len(batch))
