from notebooks.utils import (
    setup_tree_sitter,
    process_repository,
    get_embeddings,
    QdrantClient,
    initialize_qdrant,
    store_chunks,
    CodeChunk,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

REPO_ROOT = "/Users/shriramsunder/Projects/IntotheDeep"


def main():
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        main_task = progress.add_task("[magenta]Processing IntoTheDeep...", total=4)

        # Step 1: Setup tree-sitter
        progress.update(
            main_task, advance=1, description="[magenta]Setting up tree-sitter..."
        )
        parser = setup_tree_sitter()

        # Step 2: Process repository
        progress.update(main_task, description="[magenta]Processing repository...")
        chunks = process_repository(REPO_ROOT, parser)
        progress.update(main_task, advance=1)

        # Step 3: Generate embeddings
        progress.update(main_task, description="[magenta]Generating embeddings...")
        # Note: get_embeddings now has its own progress bar, so we don't need to track it here
        embs = get_embeddings(chunks=chunks, batch_size=32)
        progress.update(main_task, advance=1)

        # Step 4: Initialize Qdrant and store chunks
        progress.update(
            main_task, description="[magenta]Initializing Qdrant and storing chunks..."
        )
        qclient = initialize_qdrant()
        store_chunks(client=qclient, chunks=chunks)
        progress.update(main_task, advance=1)

    console.print("[bold green]IntoTheDeep processing completed successfully!")


if __name__ == "__main__":
    main()
