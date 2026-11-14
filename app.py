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
        ("2. Document Loading", "The app reads paired JPEG/Pascal VOC XML training records for the selected country from the official Figshare ZIP using HTTP byte-range requests, and extracts text from uploaded PDFs/TXT files."),
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
        f"- Official RDD2022 repository and dataset details: [{DATASET_SOURCE}]({DATASET_SOURCE})\n"
        "- The public Figshare ZIP is accessed with HTTP byte-range requests; only selected country training files are fetched when the user builds the app.\n"
        "- The app uses a capped sample of annotated training image/XML pairs, not the entire 47,420-image collection.\n"
        "- Test images without labels are not used as labelled retrieval evidence.\n"
        "- The image labels are ground-truth dataset annotations, not predictions made by a newly trained computer-vision model.\n"
        "- Graph links indicate annotation/category or document/category relationships; they do not establish geographic or incident-level links between different images."
    )
    st.markdown("### Runtime and storage")
    st.write(
        "The combined Figshare ZIP is not downloaded in full; only selected XML/JPEG byte ranges are fetched and held temporarily in memory while parsing. "
        "The Chroma collection and graph are held in server memory for the current running app session. "
        "No dataset archive or vector database is written to the user's computer or committed to GitHub. "
        "A Streamlit restart clears the in-memory index, so the user may need to build it again."
