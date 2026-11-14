"""RoadGraph AI: Streamlit GraphRAG application over RDD2022.

The app downloads a selected official RDD2022 country archive to the Streamlit
server's temporary directory, reads a bounded sample of annotated training
images/XML files, builds a text corpus, embeds chunks in an in-memory Chroma
vector database, constructs a NetworkX knowledge graph, and answers questions
using retrieved context plus graph-neighbour evidence.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import uuid
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import PurePosixPath
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# App configuration and dataset registry
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="RoadGraph AI | GraphRAG for Road Maintenance",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_SOURCE = "https://github.com/sekilab/RoadDamageDetector"
DAMAGE_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

# These direct country archives are linked from the official CRDDC/RDD2022
# repository. Very large archives are deliberately omitted from the UI.
COUNTRY_ARCHIVES = {
    "Czech Republic (245 MB)": {
        "country": "Czech Republic",
        "url": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip",
        "approx_mb": 245.2,
    },
    "India (502 MB)": {
        "country": "India",
        "url": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip",
        "approx_mb": 502.3,
    },
    "United States (424 MB)": {
        "country": "United States",
        "url": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip",
        "approx_mb": 423.8,
    },
    "China MotorBike (183 MB)": {
        "country": "China MotorBike",
        "url": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip",
        "approx_mb": 183.1,
    },
    "China Drone (153 MB)": {
        "country": "China Drone",
        "url": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_Drone.zip",
        "approx_mb": 152.8,
    },
}

# Compact domain reference documents form the initial road-maintenance
# knowledge base. They are explanatory project references, not official
# engineering specifications or a substitute for a local road authority.
DOMAIN_KNOWLEDGE = [
    {
        "id": "KB_D00",
        "title": "Longitudinal crack reference",
        "text": (
            "A longitudinal crack is a crack that generally runs parallel to the road's "
            "centre line or direction of travel. In RDD2022 it is represented by class D00. "
            "An inspection record should note the crack length, width, continuity, location "
            "and whether water is entering the pavement. Maintenance teams generally inspect "
            "the cause and severity before selecting an appropriate sealing or repair method."
        ),
        "categories": ["D00"],
    },
    {
        "id": "KB_D10",
        "title": "Transverse crack reference",
        "text": (
            "A transverse crack generally crosses the road perpendicular to the direction "
            "of travel. RDD2022 classifies this damage as D10. Inspection should record the "
            "extent, spacing, width and severity of the crack and check whether water or "
            "surface movement may be contributing to deterioration. Repair decisions depend "
            "on the observed condition and the applicable road authority procedure."
        ),
        "categories": ["D10"],
    },
    {
        "id": "KB_D20",
        "title": "Alligator cracking reference",
        "text": (
            "Alligator cracking, also called fatigue cracking, appears as interconnected "
            "cracks that form small polygon-like blocks resembling alligator skin. It is "
            "represented by class D20 in RDD2022. It can indicate repeated traffic loading "
            "or weakness in pavement layers. A maintenance assessment should consider the "
            "extent and severity; localised surface treatment may not be sufficient where "
            "structural failure is present."
        ),
        "categories": ["D20"],
    },
    {
        "id": "KB_D40",
        "title": "Pothole reference",
        "text": (
            "A pothole is a bowl-shaped cavity or loss of pavement material at the road "
            "surface. It is represented by class D40 in RDD2022. An inspection should record "
            "the approximate size, depth, location and severity, and assess possible water "
            "ingress or pavement breakup. A maintenance response may involve temporary or "
            "permanent patching, subject to inspection and the responsible authority's standards."
        ),
        "categories": ["D40"],
    },
    {
        "id": "KB_INSPECTION",
        "title": "Road inspection and maintenance workflow",
        "text": (
            "A basic road-maintenance workflow is: identify a reported or visually detected "
            "defect; record the road segment and evidence; classify the damage; assess extent "
            "and severity; prioritise based on safety, traffic exposure and deterioration; "
            "select an appropriate repair; complete the work; and inspect the result. Image "
            "annotations support defect identification but do not alone determine repair "
            "priority, structural condition, exact location or engineering approval."
        ),
        "categories": ["D00", "D10", "D20", "D40"],
    },
]

# -----------------------------------------------------------------------------
# Dataset loading: direct archive download to server-side temporary storage
# -----------------------------------------------------------------------------


def _normalise_code(code: str) -> str:
    code = (code or "").strip().upper()
    return code if code in DAMAGE_NAMES else code or "UNKNOWN"


def _xml_to_record(xml_bytes: bytes, xml_path: str, country: str) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    filename = root.findtext("filename") or PurePosixPath(xml_path).with_suffix(".jpg").name
    width = int(root.findtext("size/width") or 0)
    height = int(root.findtext("size/height") or 0)
    objects: list[dict[str, Any]] = []

    for obj in root.findall("object"):
        raw_name = obj.findtext("name") or "UNKNOWN"
        code = _normalise_code(raw_name)
        box = obj.find("bndbox")
        bbox = None
        if box is not None:
            try:
                bbox = {
                    "xmin": int(float(box.findtext("xmin") or 0)),
                    "ymin": int(float(box.findtext("ymin") or 0)),
                    "xmax": int(float(box.findtext("xmax") or 0)),
                    "ymax": int(float(box.findtext("ymax") or 0)),
                }
            except (TypeError, ValueError):
                bbox = None
        objects.append({"code": code, "name": DAMAGE_NAMES.get(code, raw_name), "bbox": bbox})

    counts = Counter(obj["code"] for obj in objects)
    labels = sorted(counts.keys())
    if labels:
        damage_description = "; ".join(
            f"{DAMAGE_NAMES.get(code, code)} ({code}): {counts[code]} instance(s)"
            for code in labels
        )
    else:
        damage_description = "No damage object is annotated in this XML record"

    record_id = "RDD_" + re.sub(r"[^A-Za-z0-9]+", "_", PurePosixPath(filename).stem).strip("_")[:80]
    text = (
        f"RDD2022 road image annotation. Country subset: {country}. Image file: {filename}. "
        f"Annotated road condition: {damage_description}. Image dimensions: {width} by {height} pixels. "
        f"This record contains Pascal VOC bounding-box annotations from the RDD2022 training data. "
        f"Damage codes present: {', '.join(labels) if labels else 'none recorded'}."
    )
    return {
        "id": record_id,
        "title": filename,
        "text": text,
        "doc_type": "RDD2022 annotation",
        "source": "RDD2022",
        "country": country,
        "image_file": filename,
        "damage_codes": labels,
        "objects": objects,
        "width": width,
        "height": height,
        "counts": counts,
        "xml_path": xml_path,
    }


def _draw_annotated_preview(image_bytes: bytes, objects: list[dict[str, Any]], max_side: int = 520) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    image.thumbnail((max_side, max_side))
    sx = image.width / max(original_size[0], 1)
    sy = image.height / max(original_size[1], 1)
    draw = ImageDraw.Draw(image)

    for item in objects:
        box = item.get("bbox")
        if not box:
            continue
        xy = (
            int(box["xmin"] * sx),
            int(box["ymin"] * sy),
            int(box["xmax"] * sx),
            int(box["ymax"] * sy),
        )
        draw.rectangle(xy, outline=(255, 70, 40), width=3)
        label = f"{item['code']} - {item['name']}"
        text_y = max(0, xy[1] - 16)
        draw.rectangle((xy[0], text_y, xy[0] + min(210, 7 * len(label)), text_y + 16), fill=(255, 70, 40))
        draw.text((xy[0] + 2, text_y + 1), label, fill=(255, 255, 255))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    return output.getvalue()


def download_and_parse_rdd2022(
    archive_label: str,
    max_images: int,
    progress_callback=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Download one country archive and parse a bounded sample of annotated images.

    The ZIP is written only to the Streamlit server's temporary directory and
    deleted after extraction. It is never downloaded to the user's computer.
    """
    config = COUNTRY_ARCHIVES[archive_label]
    country = config["country"]
    temp_path = None
    response = None
    try:
        response = requests.get(
            config["url"],
            stream=True,
            timeout=(30, 180),
            headers={"User-Agent": "RoadGraphAI-Student-Project/1.0"},
        )
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length") or 0)
        max_bytes = 650 * 1024 * 1024
        if total_bytes and total_bytes > max_bytes:
            raise ValueError(
                f"Selected archive is {total_bytes / (1024**2):.1f} MB, above this app's 650 MB safety limit."
            )

        with tempfile.NamedTemporaryFile(prefix="roadgraph_rdd_", suffix=".zip", delete=False) as temp_file:
            temp_path = temp_file.name
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError("Download exceeded the app's 650 MB safety limit.")
                temp_file.write(chunk)
                if progress_callback and total_bytes:
                    progress_callback(min(downloaded / total_bytes, 1.0))

        if progress_callback:
            progress_callback(1.0)

        docs: list[dict[str, Any]] = []
        previews: list[dict[str, Any]] = []
        total_class_counts: Counter = Counter()

        with zipfile.ZipFile(temp_path, "r") as archive:
            all_infos = [info for info in archive.infolist() if not info.is_dir()]
            images_by_stem: dict[str, zipfile.ZipInfo] = {}
            images_by_basename: dict[str, list[zipfile.ZipInfo]] = {}
            for info in all_infos:
                suffix = PurePosixPath(info.filename).suffix.lower()
                if suffix in {".jpg", ".jpeg"}:
                    stem_key = str(PurePosixPath(info.filename).with_suffix("")).lower()
                    images_by_stem[stem_key] = info
                    images_by_basename.setdefault(PurePosixPath(info.filename).stem.lower(), []).append(info)

            xml_infos = [info for info in all_infos if PurePosixPath(info.filename).suffix.lower() == ".xml"]
            xml_infos.sort(key=lambda info: info.filename.lower())
            selected_pairs = []
            for xml_info in xml_infos:
                stem_key = str(PurePosixPath(xml_info.filename).with_suffix("")).lower()
                image_info = images_by_stem.get(stem_key)
                if image_info is None:
                    basename = PurePosixPath(xml_info.filename).stem.lower()
                    matches = images_by_basename.get(basename, [])
                    if len(matches) == 1:
                        image_info = matches[0]
                if image_info is not None:
                    selected_pairs.append((xml_info, image_info))
                if len(selected_pairs) >= max_images:
                    break

            if not selected_pairs:
                raise RuntimeError(
                    "No matching JPEG/XML annotation pairs were found in this archive. "
                    "The archive structure may have changed; try another country subset."
                )

            for index, (xml_info, image_info) in enumerate(selected_pairs):
                xml_bytes = archive.read(xml_info)
                record = _xml_to_record(xml_bytes, xml_info.filename, country)
                # Avoid collisions from repeated image basenames in different folders.
                record["id"] = f"RDD_{country.replace(' ', '_')}_{index:05d}_{record['id'][-50:]}"
                record["archive_image_path"] = image_info.filename
                docs.append(record)
                total_class_counts.update(record["counts"])

                # Keep only a small gallery in session memory; annotations for all
                # selected records are still incorporated in the RAG corpus/graph.
                if len(previews) < 12:
                    image_bytes = archive.read(image_info)
                    try:
                        preview_bytes = _draw_annotated_preview(image_bytes, record["objects"])
                        previews.append(
                            {
                                "id": record["id"],
                                "filename": record["title"],
                                "country": country,
                                "damage": ", ".join(
                                    f"{DAMAGE_NAMES.get(code, code)} ({code})" for code in record["damage_codes"]
                                ) or "No damage annotation",
                                "bytes": preview_bytes,
                            }
                        )
                    except Exception:
                        # If a single image is malformed, retain its annotation record.
                        pass

        stats = {
            "country": country,
            "archive_label": archive_label,
            "records_loaded": len(docs),
            "class_counts": dict(total_class_counts),
            "archive_mb": config["approx_mb"],
            "source_url": config["url"],
        }
        return docs, previews, stats

    finally:
        if response is not None:
            response.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

# -----------------------------------------------------------------------------
# Supplemental document loading and RAG corpus preparation
# -----------------------------------------------------------------------------


def read_uploaded_documents(uploaded_files) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file in uploaded_files or []:
        name = file.name
        suffix = Path(name).suffix.lower()
        try:
            if suffix == ".txt":
                text = file.getvalue().decode("utf-8", errors="ignore")
            elif suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(file.getvalue()))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                continue
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                records.append(
                    {
                        "id": f"UPLOAD_{uuid.uuid4().hex[:10]}",
                        "title": name,
                        "text": text,
                        "doc_type": "Uploaded document",
                        "source": name,
                        "country": "Not specified",
                        "image_file": "",
                        "damage_codes": [],
                        "objects": [],
                        "counts": Counter(),
                    }
                )
        except Exception as exc:
            st.warning(f"Could not read {name}: {exc}")
    return records


def make_corpus(rdd_records: list[dict[str, Any]], uploaded_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus = []
    for item in DOMAIN_KNOWLEDGE:
        corpus.append(
            {
                "id": item["id"],
                "title": item["title"],
                "text": item["text"],
                "doc_type": "Domain knowledge",
                "source": "RoadGraph AI domain reference",
                "country": "General",
                "image_file": "",
                "damage_codes": list(item["categories"]),
                "objects": [],
                "counts": Counter(),
            }
        )
    corpus.extend(rdd_records)
    corpus.extend(uploaded_records)
    return corpus


def chunk_corpus(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=760,
        chunk_overlap=110,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict[str, Any]] = []
    for record in corpus:
        text = record.get("text", "").strip()
        if not text:
            continue
        parts = splitter.split_text(text)
        for part_index, part in enumerate(parts):
            chunks.append(
                {
                    "id": f"{record['id']}_CH{part_index:03d}",
                    "text": part,
                    "metadata": {
                        "source_id": str(record["id"]),
                        "title": str(record.get("title", ""))[:450],
                        "doc_type": str(record.get("doc_type", "Document")),
                        "source": str(record.get("source", ""))[:450],
                        "country": str(record.get("country", "Not specified")),
                        "damage_codes": ",".join(record.get("damage_codes", [])),
                    },
                }
            )
    return chunks

# -----------------------------------------------------------------------------
# Vector database and Knowledge Graph
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """A server-memory Chroma client; it does not persist data to user devices."""
    return chromadb.EphemeralClient()


def build_vector_database(chunks: list[dict[str, Any]]):
    client = get_chroma_client()
    collection_name = f"roadgraph_{uuid.uuid4().hex[:12]}"
    embedding_function = DefaultEmbeddingFunction()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    # Chroma metadata values must be scalar types.
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection, collection_name


def build_knowledge_graph(corpus: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for record in corpus:
        doc_id = str(record["id"])
        doc_type = record.get("doc_type", "Document")
        kind = "image_record" if doc_type == "RDD2022 annotation" else (
            "domain_document" if doc_type == "Domain knowledge" else "uploaded_document"
        )
        graph.add_node(
            doc_id,
            label=str(record.get("title", doc_id)),
            kind=kind,
            country=str(record.get("country", "")),
            doc_type=doc_type,
        )

        codes = list(record.get("damage_codes", []))
        if not codes:
            # Link documents to categories discussed in their text, while keeping
            # the relationship explicitly semantic rather than incident-level.
            text_lower = record.get("text", "").lower()
            for code, category in DAMAGE_NAMES.items():
                words = [category.lower()]
                if code == "D20":
                    words.append("fatigue cracking")
                if any(word in text_lower for word in words):
                    codes.append(code)

        for code in sorted(set(codes)):
            category_name = DAMAGE_NAMES.get(code, code)
            category_node = f"CATEGORY_{code}"
            graph.add_node(category_node, label=category_name, kind="damage_category", code=code)
            relation = "ANNOTATED_AS" if doc_type == "RDD2022 annotation" else "DISCUSSES"
            graph.add_edge(doc_id, category_node, relation=relation)
    return graph


def graph_evidence_for_hits(graph: nx.Graph, metadatas: list[dict[str, Any]]) -> list[str]:
    evidence = []
    seen = set()
    for metadata in metadatas:
        source_id = metadata.get("source_id", "")
        if not source_id or source_id not in graph:
            continue
        for neighbour in graph.neighbors(source_id):
            node = graph.nodes[neighbour]
            relation = graph.edges[source_id, neighbour].get("relation", "RELATED_TO")
            line = f"{source_id} --{relation}--> {node.get('label', neighbour)}"
            if line not in seen:
                evidence.append(line)
                seen.add(line)
    return evidence

# -----------------------------------------------------------------------------
# Retrieval, prompt construction and LLM / transparent fallback generation
# -----------------------------------------------------------------------------


def get_openai_key() -> str | None:
    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.getenv("OPENAI_API_KEY") or None


def retrieve_context(collection, graph: nx.Graph, query: str, top_k: int):
    result = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    hits = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        hits.append({"text": text, "metadata": metadata, "distance": distance})

    graph_lines = graph_evidence_for_hits(graph, metadatas)
    return hits, graph_lines


def generate_answer(query: str, hits: list[dict[str, Any]], graph_lines: list[str], model_name: str):
    context_blocks = []
    for index, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        label = meta.get("title") or meta.get("source_id") or f"Chunk {index}"
        context_blocks.append(f"[Evidence {index}: {label} | {meta.get('country', 'Not specified')} | {meta.get('doc_type', 'Document')}]\n{hit['text']}")
    graph_context = "\n".join(f"- {line}" for line in graph_lines) or "No direct graph-neighbour relationships were found for the retrieved chunks."
    context = "\n\n".join(context_blocks)
    prompt = f"""You are RoadGraph AI, an evidence-grounded assistant for road damage and maintenance research.
Answer the user's question using only the retrieved evidence and graph relationships below.
Do not invent road locations, incident links, severity ratings, repair completion, or engineering recommendations not present in the evidence.
If evidence is insufficient, state what is missing. Distinguish RDD2022 image annotations from general domain reference text.
Give a concise answer followed by a short 'Evidence used' section.

USER QUESTION:
{query}

RETRIEVED DOCUMENT CHUNKS:
{context}

KNOWLEDGE GRAPH NEIGHBOURS:
{graph_context}
"""

    api_key = get_openai_key()
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer only from supplied evidence. Be precise and transparent about limitations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content or "The model returned an empty response."
            return answer, "OpenAI LLM generation"
        except Exception as exc:
            return (
                "LLM generation failed, so the app is showing a retrieval-grounded fallback instead.\n\n"
                + _fallback_answer(hits, graph_lines)
                + f"\n\nTechnical detail: {type(exc).__name__}: {exc}",
                "Extractive fallback (LLM call failed)",
            )

    return _fallback_answer(hits, graph_lines), "Extractive fallback (no OPENAI_API_KEY configured)"


def _fallback_answer(hits: list[dict[str, Any]], graph_lines: list[str]) -> str:
    if not hits:
        return "I could not retrieve relevant evidence for this question. Try a road-damage class such as pothole, longitudinal crack, transverse crack or alligator crack."
    snippets = []
    for hit in hits[:4]:
        meta = hit["metadata"]
        title = meta.get("title") or meta.get("source_id") or "Retrieved record"
        snippets.append(f"**{title}:** {hit['text'][:650]}")
    graph_text = "\n".join(f"- {line}" for line in graph_lines[:12]) or "- No linked graph categories were retrieved."
    return (
        "The following is a retrieval-based summary of the available evidence. "
        "It is not an LLM-generated interpretation.\n\n"
        + "\n\n".join(snippets)
        + "\n\n**Related Knowledge Graph evidence**\n"
        + graph_text
    )

# -----------------------------------------------------------------------------
# Graph visualisation
# -----------------------------------------------------------------------------


def make_graph_figure(graph: nx.Graph, max_nodes: int = 180) -> go.Figure:
    if graph.number_of_nodes() == 0:
        return go.Figure()

    # For a very large selection, retain all category nodes and the most
    # connected document nodes so the browser remains responsive.
    if graph.number_of_nodes() > max_nodes:
        categories = [node for node, data in graph.nodes(data=True) if data.get("kind") == "damage_category"]
        documents = sorted(
            [node for node in graph.nodes if node not in categories],
            key=lambda node: graph.degree(node),
            reverse=True,
        )[: max(1, max_nodes - len(categories))]
        view_graph = graph.subgraph(categories + documents).copy()
    else:
        view_graph = graph

    positions = nx.spring_layout(view_graph, seed=19, k=0.75 / max(1, len(view_graph) ** 0.5), iterations=90)
    edge_x, edge_y = [], []
    for source, target in view_graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.1, color="rgba(130,140,150,0.55)"),
            hoverinfo="none",
            name="Relationships",
        )
    )

    style = {
        "damage_category": ("#28a66a", "Damage categories"),
        "image_record": ("#45a9d6", "RDD2022 image annotations"),
        "domain_document": ("#8b70d6", "Domain knowledge documents"),
        "uploaded_document": ("#dc9840", "Uploaded documents"),
    }
    for kind, (colour, legend) in style.items():
        nodes = [node for node, data in view_graph.nodes(data=True) if data.get("kind") == kind]
        if not nodes:
            continue
        x_values = [positions[node][0] for node in nodes]
        y_values = [positions[node][1] for node in nodes]
        labels = [view_graph.nodes[node].get("label", node) for node in nodes]
        custom = [
            f"ID: {node}<br>Type: {kind}<br>Connections: {graph.degree(node)}"
            for node in nodes
        ]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text" if kind == "damage_category" else "markers",
                text=labels if kind == "damage_category" else None,
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(size=18 if kind == "damage_category" else 10, color=colour, line=dict(width=1, color="white")),
                hovertext=custom,
                hoverinfo="text",
                name=legend,
            )
        )

    figure.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=25, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return figure

# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def render_pipeline_status(built: bool):
    steps = [
        "1. Domain knowledge base",
        "2. Document loading",
        "3. Text chunking / preprocessing",
        "4. Embedding generation",
        "5. Chroma vector storage",
        "6. User query",
        "7. Similarity search",
        "8. Top-k retrieved chunks",
        "9. Prompt construction with graph evidence",
        "10. LLM-based generation (API key required)",
    ]
    columns = st.columns(5)
    for index, label in enumerate(steps):
        with columns[index % 5]:
            st.markdown(
                f"<div style='border:1px solid #d9dee7;border-radius:10px;padding:10px;min-height:72px;'>"
                f"<div style='font-size:12px;color:#718096'>{'READY' if built and index < 5 else ('QUERY-TIME' if built else 'PENDING')}</div>"
                f"<div style='font-size:13px;font-weight:600'>{label}</div></div>",
                unsafe_allow_html=True,
            )


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


st.title("RoadGraph AI")
st.subheader("An ontology-guided GraphRAG prototype for road damage and maintenance")
st.caption("RDD2022 annotations + domain knowledge → vector retrieval + Knowledge Graph → evidence-grounded answers")

with st.sidebar:
    st.header("Build the knowledge base")
    selected_archive = st.selectbox(
        "RDD2022 country subset",
        list(COUNTRY_ARCHIVES.keys()),
        index=0,
        help="The app downloads the selected official country ZIP to temporary server storage. It does not save the archive on your computer.",
    )
    max_images = st.slider(
        "Maximum annotated images to ingest",
        min_value=30,
        max_value=400,
        value=100,
        step=10,
        help="A cap keeps memory and startup time manageable on free cloud hosting. The selected archive itself is downloaded in full.",
    )
    uploaded_files = st.file_uploader(
        "Optional domain documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Uploaded files are processed in the running app session and are not committed to GitHub.",
    )
    st.divider()
    st.subheader("LLM configuration")
    model_name = st.text_input("OpenAI chat model", value="gpt-4o-mini")
    if get_openai_key():
        st.success("OpenAI API key detected")
    else:
        st.info("No OpenAI key detected. Retrieval and graph search still work with an extractive fallback. Add OPENAI_API_KEY in Streamlit Secrets for LLM generation.")
    st.divider()
    build_clicked = st.button("Download RDD2022 & build app", type="primary", use_container_width=True)
    st.caption("Cloud storage note: the ZIP is temporary on the Streamlit server and deleted after parsing. Chroma uses in-memory storage.")

if build_clicked:
    download_bar = st.progress(0.0, text="Starting RDD2022 download...")
    try:
        with st.spinner("Downloading the selected RDD2022 archive and reading Pascal VOC XML annotations..."):
            rdd_records, image_previews, dataset_stats = download_and_parse_rdd2022(
                selected_archive,
                max_images,
                progress_callback=lambda value: download_bar.progress(value, text=f"Downloading archive: {value * 100:.0f}%"),
            )
        download_bar.progress(1.0, text="RDD2022 archive parsed.")
        uploaded_records = read_uploaded_documents(uploaded_files)
        corpus = make_corpus(rdd_records, uploaded_records)
        chunks = chunk_corpus(corpus)
        if not chunks:
            raise RuntimeError("No text chunks were produced from the loaded corpus.")

        with st.spinner("Generating embeddings and storing chunks in the Chroma vector database..."):
            collection, collection_name = build_vector_database(chunks)
        graph = build_knowledge_graph(corpus)

        st.session_state["rdd_records"] = rdd_records
        st.session_state["image_previews"] = image_previews
        st.session_state["dataset_stats"] = dataset_stats
        st.session_state["corpus"] = corpus
        st.session_state["chunks"] = chunks
        st.session_state["collection"] = collection
        st.session_state["collection_name"] = collection_name
        st.session_state["knowledge_graph"] = graph
        st.session_state["model_name"] = model_name
        st.session_state["built"] = True
        st.session_state["chat_history"] = []
        st.success("Knowledge base, vector index and Knowledge Graph built successfully.")
        safe_rerun()
    except Exception as exc:
        st.error(f"Build failed: {type(exc).__name__}: {exc}")
        st.info("Try a smaller archive such as China Drone or Czech Republic, or lower the image cap. If the download fails, retry later because the source host may throttle requests.")

built = bool(st.session_state.get("built"))
render_pipeline_status(built)

if not built:
    st.info("Choose a country subset and click **Download RDD2022 & build app** to create the corpus, vector database and Knowledge Graph.")
    overview_cols = st.columns(3)
    overview_cols[0].metric("RDD2022 images", "47,420", help="Dataset-wide count reported in the RDD2022 data article.")
    overview_cols[1].metric("Damage instances", "55,007", help="Annotated training instances reported in the RDD2022 data article.")
    overview_cols[2].metric("Damage categories", "4", help="D00, D10, D20 and D40 used by the CRDDC 2022 challenge.")
    st.markdown("### What this app will build")
    st.markdown(
        "- **Dataset ingestion:** downloads a country archive directly from the official RDD2022 distribution.\n"
        "- **Text corpus:** converts each Pascal VOC image annotation into searchable text and adds road-damage reference documents.\n"
        "- **Vector retrieval:** chunks the corpus, generates embeddings and stores them in an ephemeral Chroma vector database.\n"
        "- **Knowledge Graph:** connects RDD2022 image records and domain documents to damage-category nodes.\n"
        "- **GraphRAG chatbot:** retrieves top-k chunks, expands their graph neighbourhood and constructs an evidence-grounded prompt."
    )
    st.stop()

# Active built objects
rdd_records = st.session_state["rdd_records"]
image_previews = st.session_state["image_previews"]
dataset_stats = st.session_state["dataset_stats"]
corpus = st.session_state["corpus"]
chunks = st.session_state["chunks"]
collection = st.session_state["collection"]
graph = st.session_state["knowledge_graph"]

# Header metrics
class_counts = Counter()
for record in rdd_records:
    class_counts.update(record.get("counts", {}))
metric_cols = st.columns(5)
metric_cols[0].metric("Annotated images ingested", f"{len(rdd_records):,}")
metric_cols[1].metric("Text chunks", f"{len(chunks):,}")
metric_cols[2].metric("Vector records", f"{collection.count():,}")
metric_cols[3].metric("Graph nodes", f"{graph.number_of_nodes():,}")
metric_cols[4].metric("Graph edges", f"{graph.number_of_edges():,}")
st.caption(
    f"Dataset subset: {dataset_stats['country']} | Approximate archive size: {dataset_stats['archive_mb']:.1f} MB | "
    f"The app ingested {len(rdd_records)} XML-annotated training image records, not the entire dataset."
)

tabs = st.tabs(["Overview", "RAG Chatbot", "Knowledge Graph", "RDD2022 Explorer", "RAG Pipeline & Sources"])

# Overview tab
with tabs[0]:
    st.markdown("### Project workflow")
    st.markdown(
        "**RDD2022 source → XML annotation parsing → domain corpus → chunking → embeddings → Chroma → "
        "similarity retrieval → graph-neighbour expansion → prompt construction → answer.**"
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Damage distribution in the ingested sample")
        if class_counts:
            class_df = pd.DataFrame(
                [{"Code": code, "Damage category": DAMAGE_NAMES.get(code, code), "Instances": count} for code, count in sorted(class_counts.items())]
            )
            st.bar_chart(class_df.set_index("Damage category")["Instances"])
            st.dataframe(class_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No recognised D00/D10/D20/D40 damage labels were found in this selected sample.")
    with right:
        st.markdown("#### Knowledge base composition")
        composition = Counter(record.get("doc_type", "Document") for record in corpus)
        st.dataframe(
            pd.DataFrame([{"Document type": key, "Records": value} for key, value in composition.items()]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### RDD2022 classes")
        for code, name in DAMAGE_NAMES.items():
            st.write(f"**{code}** — {name}")

# RAG chatbot tab
with tabs[1]:
    st.markdown("### Ask a question over RDD2022 and road-maintenance knowledge")
    st.caption("The chatbot uses top-k vector retrieval plus graph-neighbour evidence. Add OPENAI_API_KEY in Streamlit Secrets to enable LLM response generation.")
    query_col, k_col = st.columns([4, 1])
    with query_col:
        query = st.text_input(
            "Natural-language question",
            placeholder="Which road damage categories are represented, and what maintenance evidence is available for potholes?",
            key="rag_query_input",
        )
    with k_col:
        top_k = st.selectbox("Top-k", [3, 4, 5, 6, 8], index=2)

    if st.button("Retrieve and answer", type="primary", key="run_rag"):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching the vector database and expanding Knowledge Graph relationships..."):
                hits, graph_lines = retrieve_context(collection, graph, query.strip(), int(top_k))
                answer, generation_mode = generate_answer(query.strip(), hits, graph_lines, model_name)
                st.session_state["last_rag_result"] = {
                    "query": query.strip(),
                    "hits": hits,
                    "graph_lines": graph_lines,
                    "answer": answer,
                    "generation_mode": generation_mode,
                }

    result = st.session_state.get("last_rag_result")
    if result:
        st.markdown("#### Answer")
        st.info(f"Generation mode: {result['generation_mode']}")
        st.markdown(result["answer"])
        st.markdown("#### Retrieved top-k chunks")
        for index, hit in enumerate(result["hits"], start=1):
            metadata = hit["metadata"]
            source_name = metadata.get("title") or metadata.get("source_id")
            distance = hit.get("distance")
            with st.expander(f"{index}. {source_name} | {metadata.get('doc_type')} | distance: {distance:.4f}" if isinstance(distance, (int, float)) else f"{index}. {source_name}"):
                st.write(hit["text"])
                st.caption(
                    f"Source ID: {metadata.get('source_id')} | Country: {metadata.get('country')} | "
                    f"Damage codes: {metadata.get('damage_codes') or 'Not specified'}"
                )
        st.markdown("#### Knowledge Graph evidence")
        if result["graph_lines"]:
            for line in result["graph_lines"]:
                st.write(f"- `{line}`")
        else:
            st.write("No direct neighbour links were found for the retrieved records.")

# Knowledge Graph tab
with tabs[2]:
    st.markdown("### Interactive Knowledge Graph")
    st.markdown(
        "Blue nodes are RDD2022 image-annotation records; green nodes are damage categories; "
        "purple nodes are built-in domain reference documents; orange nodes are user-uploaded documents."
    )
    st.plotly_chart(make_graph_figure(graph), use_container_width=True)
    edge_rows = []
    for source, target, data in graph.edges(data=True):
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        edge_rows.append(
            {
                "Source": source_data.get("label", source),
                "Relationship": data.get("relation", "RELATED_TO"),
                "Target": target_data.get("label", target),
                "Source ID": source,
                "Target ID": target,
            }
        )
    with st.expander("View graph relationships as a table"):
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Knowledge Graph (GraphML)",
        data="".join(nx.generate_graphml(graph)),
        file_name="roadgraph_ai_knowledge_graph.graphml",
        mime="application/graphml+xml",
    )

# Dataset explorer tab
with tabs[3]:
    st.markdown("### RDD2022 annotation explorer")
    st.write(
        f"Loaded **{len(rdd_records)} annotated records** from the **{dataset_stats['country']}** subset. "
        "Each record is derived from a Pascal VOC XML annotation paired with an image in the archive."
    )
    rows = []
    for record in rdd_records:
        rows.append(
            {
                "Image": record["title"],
                "Country subset": record["country"],
                "Damage codes": ", ".join(record["damage_codes"]) or "No annotated object",
                "Damage labels": ", ".join(DAMAGE_NAMES.get(code, code) for code in record["damage_codes"]) or "No annotated object",
                "Objects": len(record["objects"]),
                "Image size": f"{record['width']} × {record['height']}" if record["width"] and record["height"] else "Unknown",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("#### Annotated image samples")
    if image_previews:
        columns = st.columns(3)
        for index, preview in enumerate(image_previews):
            with columns[index % 3]:
                st.image(preview["bytes"], use_container_width=True)
                st.caption(f"{preview['filename']}\n{preview['damage']}")
    else:
        st.info("No image previews could be rendered, but XML annotation records were loaded.")
    st.caption("Bounding boxes shown in previews are read from the dataset's original XML annotations; they are not model predictions.")

# Pipeline and source tab
with tabs[4]:
    st.markdown("### Mapping the implementation to the 10 RAG stages")
    pipeline_rows = [
        ("1. Domain Documents / Knowledge Base", "Built-in road-damage reference records for D00, D10, D20 and D40, plus optional user PDF/TXT files."),
        ("2. Document Loading", "The app downloads an official RDD2022 country archive on the server, reads paired JPEG/Pascal VOC XML training records, and extracts text from uploaded PDFs/TXT files."),
        ("3. Text Chunking and Pre-processing", "LangChain RecursiveCharacterTextSplitter creates 760-character chunks with 110-character overlap."),
        ("4. Embedding Generation", "Chroma's default embedding function generates dense all-MiniLM-L6-v2 sentence embeddings using ONNX Runtime."),
        ("5. Vector Database Storage", "Chunks, embeddings and metadata are stored in an ephemeral in-memory Chroma collection."),
        ("6. User Query", "The user enters a natural-language road-damage or maintenance question in the chatbot."),
        ("7. Similarity Search / Retrieval", "Chroma performs cosine-distance vector retrieval over the embedded chunks."),
        ("8. Top-k Relevant Chunks", "The interface displays the selected top-k text chunks, source metadata and vector distance."),
        ("9. Prompt Construction using Retrieved Context", "The retrieved chunks are combined with one-hop Knowledge Graph relationships to construct an evidence-grounded prompt."),
        ("10. LLM-based Generation", "When OPENAI_API_KEY is configured, the app calls the selected OpenAI chat model. Without a key, it shows a clearly labelled extractive retrieval fallback."),
    ]
    for title, description in pipeline_rows:
        with st.expander(title, expanded=False):
            st.write(description)

    st.markdown("### Data source and scope")
    st.markdown(
        f"- Official RDD2022 repository and country archive links: [{DATASET_SOURCE}]({DATASET_SOURCE})\n"
        "- The selected archive is downloaded only when the user clicks the build button.\n"
        "- The app uses a capped sample of annotated training image/XML pairs, not the entire 47,420-image collection.\n"
        "- Test images without labels are not used as labelled retrieval evidence.\n"
        "- The image labels are ground-truth dataset annotations, not predictions made by a newly trained computer-vision model.\n"
        "- Graph links indicate annotation/category or document/category relationships; they do not establish geographic or incident-level links between different images."
    )
    st.markdown("### Runtime and storage")
    st.write(
        "The ZIP archive is temporarily downloaded to the Streamlit server's temporary directory and deleted after parsing. "
        "The Chroma collection and graph are held in server memory for the current running app session. "
        "No dataset archive or vector database is written to the user's computer or committed to GitHub. "
        "A Streamlit restart clears the in-memory index, so the user may need to build it again."
    )

st.divider()
st.caption("RoadGraph AI | Academic prototype. Dataset attribution: Arya et al., RDD2022. Not an operational road-inspection or engineering decision system.")
