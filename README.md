# RoadGraph AI — Streamlit GraphRAG for RDD2022

**Project title:** *RoadGraph AI: An Ontology-Guided Multimodal GraphRAG System for Intelligent Road Maintenance*

RoadGraph AI is a cloud-hosted academic prototype that turns road-damage image annotations and road-maintenance reference text into a searchable corpus, an embedding-based vector index, and a visual Knowledge Graph. A user can ask natural-language questions and inspect both the retrieved text evidence and the connected graph concepts.

The application is designed for **Streamlit Community Cloud**. The RDD2022 archive is downloaded by the running Streamlit server when the user clicks the build button. It is not necessary to download the dataset to your personal computer or commit it to GitHub.

---

## 1. What the application does

1. Downloads a selected country-specific RDD2022 archive from the official dataset distribution.
2. Reads paired road images (`.jpg`) and Pascal VOC annotation files (`.xml`) from the annotated training set.
3. Converts each XML annotation into a text record containing its image name, country subset, damage codes, class counts, and bounding-box information.
4. Adds a small built-in road-damage reference knowledge base and optional user-supplied PDF/TXT documents.
5. Chunks the text using LangChain's `RecursiveCharacterTextSplitter`.
6. Generates dense text embeddings using Chroma's default `all-MiniLM-L6-v2` embedding function (ONNX Runtime).
7. Stores chunk text, embeddings, and metadata in an **in-memory Chroma vector database**.
8. Builds a NetworkX Knowledge Graph linking each image/document record to road-damage category nodes.
9. Retrieves the top-k chunks for a user question and expands the corresponding graph neighbourhood.
10. Constructs an evidence-grounded prompt and calls an OpenAI chat model when an API key is configured. Without a key, the app shows a clearly labelled extractive retrieval fallback.

## 2. Dataset used: RDD2022

RDD2022 is a multi-national road-damage image dataset released through the Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022). The dataset article reports **47,420 road images** and **55,007 annotated damage instances**, covering six country subsets: India, Japan, Czech Republic, Norway, United States, and China. The challenge's four main damage categories are:

| Code | Damage category | Description |
|---|---|---|
| D00 | Longitudinal Crack | Crack generally running along the direction of travel |
| D10 | Transverse Crack | Crack generally crossing the direction of travel |
| D20 | Alligator Crack | Interconnected fatigue-crack pattern |
| D40 | Pothole | Bowl-shaped cavity or loss of pavement material |

The challenge release provides annotated training images and XML labels in Pascal VOC format. Test images are supplied without annotations; this prototype uses the labelled XML/image pairs from the training data as retrieval evidence.

### Official links

- [RDD2022 dataset/repository and country archive links](https://github.com/sekilab/RoadDamageDetector)
- [CRDDC 2022 dataset information and category definitions](https://crddc2022.sekilab.global/data/)
- [RDD2022 data article (Geoscience Data Journal)](https://doi.org/10.1002/gdj3.260)
- [RDD2022 Figshare record](https://doi.org/10.6084/m9.figshare.21431547.v1)

The RDD2022 repository lists country archive sizes. To keep the app within common cloud-resource limits, it exposes these smaller subsets:

| App choice | Approximate archive size |
|---|---:|
| Czech Republic | 245.2 MB |
| India | 502.3 MB |
| United States | 423.8 MB |
| China MotorBike | 183.1 MB |
| China Drone | 152.8 MB |

The default is **Czech Republic** because it is a more manageable starting point. Select **India** in the sidebar to use the India subset. The app intentionally does not offer the much larger Japan and Norway archives in its country selector.

**Important:** The application downloads the complete selected country ZIP archive, but only parses the user-selected maximum number of annotated image/XML pairs (30–400). This is a bounded prototype, not ingestion of all 47,420 images.

## 3. Project structure

```text
RoadGraphAI_Streamlit/
│
├── app.py                  # Complete Streamlit application and RAG pipeline
├── requirements.txt        # Python packages
├── runtime.txt             # Python runtime for Streamlit Community Cloud
├── README.md               # Project explanation and setup instructions
├── .gitignore              # Excludes secrets and temporary files
└── .streamlit/
    └── config.toml         # Streamlit theme and app configuration
```

No dataset archive, model checkpoint, or local vector-store folder is required in GitHub.

## 4. RAG architecture — all 10 stages

| Stage | What happens in this project | Implementation |
|---|---|---|
| 1. Domain Documents / Knowledge Base | Built-in explanatory references for D00, D10, D20, D40 and road inspection; optional PDF/TXT uploads | `DOMAIN_KNOWLEDGE`, `read_uploaded_documents()` |
| 2. Document Loading | Download RDD2022 ZIP, read matching JPEG/XML pairs, parse damage labels and boxes | `download_and_parse_rdd2022()`, `_xml_to_record()` |
| 3. Text Chunking and Pre-processing | Clean text and split into overlapping chunks | `chunk_corpus()` + `RecursiveCharacterTextSplitter` |
| 4. Embedding Generation | Convert each text chunk into a dense vector representation | Chroma `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2`) |
| 5. Vector Database Storage | Store text, embeddings and source metadata in an ephemeral vector collection | `build_vector_database()` + Chroma `EphemeralClient` |
| 6. User Query | User enters a question in the chatbot | RAG Chatbot tab |
| 7. Similarity Search / Retrieval | Find semantically similar chunks using cosine distance | `retrieve_context()` + `collection.query()` |
| 8. Top-k Relevant Chunks | Select and display the requested number of relevant chunks | `top_k` control in RAG Chatbot |
| 9. Prompt Construction using Retrieved Context | Combine retrieved chunks and one-hop Knowledge Graph evidence | `generate_answer()` |
| 10. LLM-based Generation | Generate an answer from the evidence using an OpenAI chat model | OpenAI Chat Completions; extractive fallback if no key is configured |

### How GraphRAG is added to conventional RAG

Conventional RAG uses the retrieved text chunks as context. RoadGraph AI additionally uses the source IDs of those chunks to look up their connected Knowledge Graph nodes. These graph-neighbour relationships are added to the prompt as extra evidence.

Example graph pattern:

```text
RDD2022 image annotation record
              |
         ANNOTATED_AS
              |
       Pothole (D40)
```

A domain reference document can also connect to one or more damage-category nodes using a `DISCUSSES` relationship. This lets the application show relationships between image annotations and general road-maintenance knowledge without falsely asserting that two different images depict the same physical road incident.

## 5. Knowledge Graph design

The graph is constructed programmatically with NetworkX.

**Node types:**

- **RDD2022 image record:** one annotated image/XML pair.
- **Damage category:** D00, D10, D20 or D40.
- **Domain document:** a built-in road-damage or inspection reference.
- **Uploaded document:** an optional PDF or TXT supplied through the app.

**Relationship types:**

- `ANNOTATED_AS`: an RDD2022 image record is labelled with a damage class in its XML annotation.
- `DISCUSSES`: a domain or uploaded document discusses a damage category.

The Knowledge Graph tab displays the graph interactively and allows export as GraphML.

## 6. Cloud deployment — GitHub to Streamlit Community Cloud

### Step A — Upload this project to GitHub

1. Create a new GitHub repository, for example `RoadGraphAI-GraphRAG`.
2. Upload the contents of this project folder to the repository root. The root must contain `app.py`, `requirements.txt`, `runtime.txt`, `README.md`, and `.streamlit/config.toml`.
3. Do **not** upload the RDD2022 ZIP archive, API keys, `.env` files, or `secrets.toml`.

### Step B — Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with GitHub and choose **Create app**.
3. Select the repository, branch, and main file path `app.py`.
4. Deploy the app. Streamlit installs packages from `requirements.txt`.
5. Wait for the initial application build to finish. The first embedding-model use may also download Chroma's ONNX embedding model to the cloud runtime.

### Step C — Add the OpenAI API key (for LLM generation)

The application can run retrieval, vector search and Knowledge Graph exploration without an OpenAI key. To enable the LLM answer-generation stage:

1. Open the deployed app's **Manage app → Settings → Secrets** section.
2. Add the following TOML secret:

```toml
OPENAI_API_KEY = "your_openai_api_key"
```

3. Save the secret and restart/reboot the app if needed.

The key must be kept in Streamlit Secrets. Never put it in `app.py`, GitHub, README, or a public commit. The default chat model is `gpt-4o-mini`; it can be changed in the app's sidebar if your API account supports another model.

### Step D — Build the data and graph

1. Open the deployed app.
2. In the sidebar, select **Czech Republic** for a smaller first run or **India** for the Indian RDD2022 subset.
3. Set the maximum annotated images to ingest. Start with 50–100 for a first demonstration.
4. Optionally upload road-maintenance PDF/TXT reference documents.
5. Click **Download RDD2022 & build app**.
6. Wait for the archive download, XML parsing, chunking, embedding and graph construction to finish.
7. Explore the Overview, RAG Chatbot, Knowledge Graph, RDD2022 Explorer, and RAG Pipeline tabs.

Example questions:

- `What road damage categories are represented in the selected RDD2022 subset?`
- `Explain the pothole class D40 and what inspection details should be recorded.`
- `Which annotated images in the ingested sample contain alligator cracking?`
- `How does the Knowledge Graph connect image annotations to road-damage concepts?`

## 7. No personal computer storage required — what that means

- The user's browser only displays the interface. The data archive is downloaded by the Streamlit server, not by the user's PC.
- The ZIP is written temporarily to the cloud server's temporary directory so Python can read its internal XML and image files. The app deletes the ZIP after parsing.
- The Chroma vector database is in-memory (`EphemeralClient`), not a persistent local database.
- Image previews and graph objects remain in the running Streamlit session's server memory.
- GitHub stores only source code and configuration files.
- Streamlit Community Cloud can restart or hibernate apps. When the process restarts, the in-memory vector index and session state are cleared; the user must build the knowledge base again.

This is **no local storage on your personal computer**, not a claim that the application uses zero server-side temporary memory/storage. Streamlit Cloud also has resource limits, so a capped sample is used. The selected archive still downloads in full before the sample is parsed.

## 8. Evaluation and demonstration notes

The current application demonstrates the full RAG workflow and the graph-augmented evidence path. During a viva/demo, show:

1. The selected RDD2022 country and number of XML/image pairs ingested.
2. The damage-category distribution for the selected sample.
3. The RAG Chatbot query, top-k retrieved chunks and source metadata.
4. The Knowledge Graph showing image records connected to damage-category nodes.
5. One or more graph relationships in the relationship table.
6. The RAG Pipeline tab, which maps the implementation to all 10 architecture stages.

The app currently has no separately trained road-damage detection model. It uses the **ground-truth bounding boxes and class names supplied by RDD2022 XML annotations**. Its purpose is document/annotation retrieval and graph-based exploration, not new visual inference on arbitrary uploaded images.

## 9. Limitations (state these clearly)

- Only a capped sample from one selected country archive is ingested per build. It is not the entire 47,420-image dataset.
- The application must download the complete selected country ZIP before parsing a sample. A large archive or slow source host can cause a timeout or resource issue.
- The Knowledge Graph is an ontology-guided prototype with explicit document-to-category relationships. It is not a fully learned ontology or an LLM-extracted graph.
- This release uses NetworkX for graph storage and visualisation. It does not currently include a Neo4j database, a native Cypher execution console, LangChain `LLMGraphTransformer`, or LlamaIndex `KnowledgeGraphIndex`; do not claim these components are implemented.
- Graph edges do not represent geographic adjacency, shared road-segment identity, or proof that two images show the same incident.
- RDD2022 class labels and boxes are treated as the dataset's annotations; the app does not independently verify annotation quality.
- LLM generation requires a valid OpenAI API key and may incur API charges. Without it, the app uses a transparent extractive fallback.
- An API-generated answer is grounded by prompt instructions but should still be checked against the displayed retrieved evidence.
- Cloud restarts clear in-memory Chroma and app state; this prototype has no durable external vector database.

## 10. Citation

Please cite the dataset if you use it in the project report:

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2024). **RDD2022: A multi-national image dataset for automatic road damage detection.** *Geoscience Data Journal, 11*(4), 846–862. https://doi.org/10.1002/gdj3.260

Dataset record: **RDD2022 – The multi-national Road Damage Dataset released through CRDDC'2022.** Figshare. https://doi.org/10.6084/m9.figshare.21431547.v1

---

## Viva explanation in one paragraph

RoadGraph AI is a GraphRAG application for road-maintenance knowledge retrieval. It downloads a selected annotated subset of the RDD2022 dataset, parses the Pascal VOC XML labels associated with road images, and converts each annotation into searchable text. These records are combined with road-damage reference documents, split into overlapping chunks, embedded and stored in a Chroma vector database. A NetworkX Knowledge Graph links each image/document record with its damage category, such as pothole or alligator cracking. When a user asks a question, the system retrieves the top-k relevant chunks, expands their connected graph relationships, builds a prompt from both sources of evidence, and generates an answer using an OpenAI model when configured. The app also displays the annotated image samples, damage statistics, Knowledge Graph, retrieval evidence and all ten RAG stages.
