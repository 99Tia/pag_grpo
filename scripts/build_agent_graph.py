from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ppr_agent.graph_builder import (  # noqa: E402
    GraphBuilderConfig,
    PPRAgentGraphBuilder,
)

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build entity-passage graph from OpenIE results."
    )

    parser.add_argument(
        "--openie_path",
        type=str,
        required=True,
        help="Path to OpenIE JSON file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where graph.pickle and graph_metadata.json will be saved.",
    )

    parser.add_argument(
        "--directed",
        action="store_true",
        help="Build directed graph. Default is undirected.",
    )

    parser.add_argument(
        "--include_ner_entities",
        action="store_true",
        help=(
            "Also connect NER-only entities to passages. "
            "Default is False because NER-only entities can add noise."
        ),
    )

    parser.add_argument(
        "--relation_edge_weight",
        type=float,
        default=1.0,
        help="Weight for entity-entity relation edges.",
    )

    parser.add_argument(
        "--passage_entity_edge_weight",
        type=float,
        default=1.0,
        help="Weight for passage-entity context edges.",
    )

    parser.add_argument(
        "--graph_filename",
        type=str,
        default="graph.pickle",
    )

    parser.add_argument(
        "--metadata_filename",
        type=str,
        default="graph_metadata.json",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not os.path.exists(args.openie_path):
        raise FileNotFoundError(f"OpenIE file not found: {args.openie_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    config = GraphBuilderConfig(
        output_dir=args.output_dir,
        graph_filename=args.graph_filename,
        metadata_filename=args.metadata_filename,
        directed=args.directed,
        relation_edge_weight=args.relation_edge_weight,
        passage_entity_edge_weight=args.passage_entity_edge_weight,
        include_ner_entities=args.include_ner_entities,
    )

    logger.info("Building graph.")
    logger.info("OpenIE path: %s", args.openie_path)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Directed: %s", args.directed)
    logger.info("Include NER-only entities: %s", args.include_ner_entities)

    builder = PPRAgentGraphBuilder(config)
    result = builder.build_from_openie_file(args.openie_path)

    print("\nGraph build result:")
    print(json.dumps(asdict(result), indent=2))

    metadata_path = os.path.join(args.output_dir, args.metadata_filename)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("\nGraph metadata summary:")
    print(json.dumps(
        {
            "num_entity_nodes": metadata.get("num_entity_nodes"),
            "num_passage_nodes": metadata.get("num_passage_nodes"),
            "num_total_nodes": metadata.get("num_total_nodes"),
            "num_total_edges": metadata.get("num_total_edges"),
            "num_unique_triples": metadata.get("num_unique_triples"),
        },
        indent=2,
    ))

    print("\nSaved files:")
    print("graph:", os.path.join(args.output_dir, args.graph_filename))
    print("metadata:", metadata_path)


if __name__ == "__main__":
    main()