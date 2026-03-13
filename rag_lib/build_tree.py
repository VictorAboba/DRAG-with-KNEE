from pathlib import Path
import json
import uuid

from qdrant_client.models import PointStruct, Document

from .clients import RAGalicClient
from .chunking import chunk_document
from .utils import llm_call
from .dataschemes import DescriptorOutput, Node

DESCRIPTOR_SYSTEM_PROMPT = """# Instruction
You are an expert in analyzing technical and regulatory documentation. Your task is to generate metadata for a document tree structure based on the provided text.

## Rules
- **Description**: Formulate a brief technical summary (3-5 sentences). This should be a complete definition combining the section's theme. Use professional terminology, avoid introductory words.
- **Keywords**: Highlight 5-10 unique anchor terms in their base form. Include: specific details, abbreviations, synonyms, and English equivalents.
- **Aggregation**: If descriptions of several subsections are provided as input, create a single generalizing description for the parent node.

## Constraints
1. Do not duplicate the fragment text verbatim.
2. If the fragment contains no meaning (empty/artifacts), provide a general description based on the document context.

## Response format
You must return ONLY a valid JSON object. Do not include markdown blocks like ```json. Do not use double quotes inside string values unless escaped.
Generate a JSON object with the following fields:
- `description`: Technical summary of the context (for semantic search).
- `keywords`: List of keywords and abbreviations (for exact search).
"""

PATH_TO_PARSED_DOCS = Path(__file__).parent / "database" / "parsed_files"


def build_tree(path: Path, width: int = 3, batch_size: int = 5):
    name = path.name
    print(f"Building tree for: {name}")
    chunks = chunk_document(path)

    if len(chunks) == 0:
        print(f"No valid chunks found for {name}. Skipping.")
        raise ValueError(f"No valid chunks found for {name}.")

    with open(
        PATH_TO_PARSED_DOCS / f"{path.with_suffix('.json').name}", "w", encoding="utf-8"
    ) as f:
        json.dump([""] + chunks, f, ensure_ascii=False, indent=4)

    with RAGalicClient() as client:
        cur_id = client.client.count("ragalic").count

    nodes: list[Node] = []
    for i, chunk in enumerate(chunks, start=1):
        messages = [
            {"role": "system", "content": chunk},
            {"role": "user", "content": f"Content to describe:\n{chunk}"},
        ]
        retries = 0
        while retries < 3:
            try:
                output, _ = llm_call(messages, DescriptorOutput)
                output = DescriptorOutput.model_validate_json(output)
                break
            except Exception as e:
                print(
                    f"Error during LLM call (for desciption): {e}. Retrying ({retries + 1}/3)..."
                )
                retries += 1
        else:
            print(
                "Failed to get a valid response after 3 attempts. Using default values."
            )
            output = DescriptorOutput()
        node = Node(
            id=cur_id,
            file_name=name,
            parent_id=None,
            child_ids=[],
            description=output.description,
            keywords=output.keywords,
            page_start=i,
            page_end=i,
        )
        nodes.append(node)
        cur_id += 1

    def create_parent_node_and_update_children(child_nodes: list[Node], cur_id) -> Node:
        combined_description = "\n".join(
            [str(node.description) for node in child_nodes]
        )
        messages = [
            {"role": "system", "content": DESCRIPTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Aggregate description for the following content:\n{combined_description}",
            },
        ]
        retries = 0
        while retries < 3:
            try:
                output, _ = llm_call(messages, DescriptorOutput)
                output = DescriptorOutput.model_validate_json(output)
                break
            except Exception as e:
                print(
                    f"Error during LLM call (for parent node description): {e}. Retrying ({retries + 1}/3)..."
                )
                retries += 1
        else:
            print(
                "Failed to get a valid response after 3 attempts. Using default values for parent node."
            )
            output = DescriptorOutput()

        start_page = min(child_nodes, key=lambda n: n.page_start).page_start
        end_page = max(child_nodes, key=lambda n: n.page_end).page_end
        all_child_keywords = list(
            set(kw for node in child_nodes for kw in node.keywords)
        )

        parent_node = Node(
            id=cur_id,
            file_name=name,
            parent_id=None,
            child_ids=[node.id for node in child_nodes],
            description=output.description,
            keywords=all_child_keywords,
            page_start=start_page,
            page_end=end_page,
        )

        for node in child_nodes:
            node.parent_id = parent_node.id

        return parent_node

    # Create parent nodes for every `width` child nodes
    nodes_to_process = nodes.copy()
    while len(nodes_to_process) > 1:
        new_nodes = []
        for i in range(0, len(nodes_to_process), width):
            child_nodes = nodes_to_process[i : i + width]
            if len(child_nodes) == 1:
                new_nodes.append(child_nodes[0])  # No need to create a parent node
            else:
                parent_node = create_parent_node_and_update_children(
                    child_nodes, cur_id
                )
                nodes.append(parent_node)
                new_nodes.append(parent_node)
                cur_id += 1
        nodes_to_process = new_nodes

    # Root should have parent_id = -1
    for node in nodes:
        if node.parent_id is None:
            node.parent_id = -1

    with RAGalicClient() as client:
        if not client.client.collection_exists("ragalic"):
            client.create_collection(
                collection_name="ragalic",
                vectors_config={
                    "dense": list(client.client.get_fastembed_vector_params().items())[
                        0
                    ][1]
                },  # Конфиг для Dense
                sparse_vectors_config={
                    "sparse": list(
                        client.client.get_fastembed_sparse_vector_params().items()
                    )[0][1]
                },  # Конфиг для Sparse
            )

        num_batches = (len(nodes) + batch_size - 1) // batch_size
        for i in range(0, len(nodes), batch_size):
            print(f"Upserting batch {i // batch_size + 1}/{num_batches} [{name}]...")
            batch = nodes[i : i + batch_size]
            points = [
                PointStruct(
                    id=node.id,
                    vector={
                        "sparse": Document(
                            text=node.get_sparse_text(),
                            model=client.client.sparse_embedding_model_name,  # type: ignore
                        ),
                        "dense": Document(
                            text=node.get_dense_text(),
                            model=client.client.embedding_model_name,  # type: ignore
                        ),
                    },
                    payload=node.model_dump(),
                )
                for node in batch
            ]
            client.client.upsert(
                collection_name="ragalic",
                points=points,
                wait=True,
            )
    
    print(f"Finished building tree for: {name}")


if __name__ == "__main__":
    build_tree(
        Path(
            r"C:\Users\admin\Desktop\RAGalic\rag_lib\database\A systematic review of computer vision-based personal protective equipment compliance in industry practice advancements, challenges and future directions.pdf"
        )
    )
